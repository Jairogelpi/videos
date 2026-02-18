import { createClient } from '@supabase/supabase-js';
import { bundle } from '@remotion/bundler';
import { renderMedia, getCompositions } from '@remotion/renderer';
import path from 'path';
import fs from 'fs';
import os from 'os';
import dotenv from 'dotenv';

dotenv.config();

// CLI Entry Point
async function main() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error('Usage: tsx cli-render.ts <payload-json-path>');
        process.exit(1);
    }

    const payloadPath = args[0];
    const payload = JSON.parse(fs.readFileSync(payloadPath, 'utf-8'));

    const { jobId, user_id, asset_id, ...rest } = payload;
    // Normalize keys: API sends snake_case or camelCase? 
    // API (route.ts) sends: job_id, asset_id, user_id
    // But Render logic in index.ts expects: jobId, analysisAssetId, userId

    // Mapping
    const effectiveJobId = jobId || payload.job_id;
    const effectiveUserId = user_id || payload.user_id;
    const analysisAssetId = asset_id || payload.asset_id || payload.analysisAssetId;

    console.log(`[${effectiveJobId}] Starting Cloud Render for user ${effectiveUserId}...`);

    await runRender(effectiveJobId, effectiveUserId, analysisAssetId);
}

// Re-implementing logic from index.ts but optimized for single-run
const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

const CALLBACK_URL = process.env.CALLBACK_URL;
const CALLBACK_TOKEN = process.env.CALLBACK_TOKEN;

async function reportSync(jobId: string, event: string, data: any, userId?: string) {
    if (!CALLBACK_URL) return;
    try {
        await fetch(CALLBACK_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${CALLBACK_TOKEN}`
            },
            body: JSON.stringify({
                jobId,
                userId,
                event,
                ...data
            })
        });
    } catch (e) {
        console.error('Failed to report status:', e);
    }
}

async function runRender(jobId: string, userId: string, analysisAssetId: string) {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'remotion-'));
    const outputPath = path.join(tmpDir, 'output.mp4');

    try {
        await reportSync(jobId, 'stage', { status: 'compositing', progress: 10 }, userId);

        // 1. Download Analysis JSON
        const { data: assetData } = await supabase.from('job_assets').select('*').eq('id', analysisAssetId).single();
        if (!assetData) throw new Error(`Analysis asset ${analysisAssetId} not found`);

        const { data: fileData, error: downloadError } = await supabase.storage.from('assets').download(assetData.path);
        if (downloadError || !fileData) {
            throw new Error(`Failed to download analysis JSON: ${downloadError?.message}`);
        }

        const analysisJson = JSON.parse(await fileData.text());

        // 1a. Fetch Audio Asset
        const { data: jobInfo } = await supabase.from('jobs').select('input_audio_asset_id').eq('id', jobId).single();
        const { data: audioAsset } = await supabase.from('job_assets').select('*').eq('id', jobInfo?.input_audio_asset_id).single();

        const audioUrl = `${process.env.SUPABASE_URL}/storage/v1/object/public/assets/${audioAsset?.path}`;
        const videoBgUrl = analysisJson.videoBgUrl && !analysisJson.videoBgUrl.startsWith('http')
            ? `${process.env.SUPABASE_URL}/storage/v1/object/public/assets/${analysisJson.videoBgUrl}`
            : analysisJson.videoBgUrl;

        const inputProps = {
            ...analysisJson,
            audioUrl,
            videoBgUrl,
            styleId: analysisJson.styleId,
            fontFamily: analysisJson.fontFamily,
            animationEffect: analysisJson.animationEffect,
            lyricColor: analysisJson.lyricColor,
            lyricOpacity: analysisJson.lyricOpacity
        };

        // 2. Render Video
        await reportSync(jobId, 'stage', { status: 'compositing', progress: 30 }, userId);

        // Adjust path to point to the actual remotion root (apps/render)
        // Since we are running from workers/render, and apps/render is usually the same or sibling?
        // Wait, looking at file structure:
        // workers/render/package.json
        // workers/render/src/index.ts
        // index.ts says: path.join(process.cwd(), '../../apps/render/src/index.ts')
        // This implies workers/render is a separate package from apps/render
        // I need to ensure the composition entry point is correct in the Cloud environment.

        // In Modal, we will copy the whole repo or just the relevant parts.
        // For simplicity, we assume we mount '../../apps/render' as well.

        const compositionId = 'LyricVideo';
        // We will mount apps/render to /root/apps/render in Modal
        const entry = path.resolve(process.cwd(), '../../apps/render/src/index.tsx');

        console.log(`[${jobId}] Bundling form ${entry}...`);
        const bundled = await bundle(entry);

        const comps = await getCompositions(bundled, { inputProps });
        const composition = comps.find((c) => c.id === compositionId);
        if (!composition) throw new Error(`Composition ${compositionId} not found`);

        await reportSync(jobId, 'stage', { status: 'encoding', progress: 50 }, userId);
        console.log(`[${jobId}] Rendering...`);

        await renderMedia({
            composition,
            serveUrl: bundled,
            codec: 'h264',
            outputLocation: outputPath,
            inputProps,
            onProgress: ({ progress }: { progress: number }) => {
                const p = Math.floor(50 + (progress * 40));
                if (p % 10 === 0) console.log(`[${jobId}] Progress: ${p}%`);
                // Trottle reports?
                // reportSync(jobId, 'stage', { status: 'encoding', progress: p }, userId);
            }
        });

        // 3. Upload Output
        await reportSync(jobId, 'stage', { status: 'encoding', progress: 95 }, userId);
        const outputKey = `${jobId}/final.mp4`;

        console.log(`[${jobId}] Uploading to Supabase...`);
        const fileBuffer = fs.readFileSync(outputPath);
        await supabase.storage.from('outputs').upload(outputKey, fileBuffer, {
            contentType: 'video/mp4',
            upsert: true
        });

        // 4. Final Callback
        await reportSync(jobId, 'asset', {
            kind: 'final_video',
            url: outputKey
        });

        await reportSync(jobId, 'stage', { status: 'completed', progress: 100 }, userId);
        console.log(`[${jobId}] Cloud Render completed!`);

    } catch (err: any) {
        console.error(`[${jobId}] Render failed:`, err);
        await reportSync(jobId, 'error', {
            message: err.message,
            errorCode: 'RENDER_FAILURE'
        }, userId);
        process.exit(1);
    } finally {
        if (fs.existsSync(tmpDir)) {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    }
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
