import { Worker, Job } from 'bullmq';
import { createClient } from '@supabase/supabase-js';
import { bundle } from '@remotion/bundler';
import { renderMedia, getCompositions } from '@remotion/renderer';
import path from 'path';
import fs from 'fs';
import os from 'os';
import dotenv from 'dotenv';

dotenv.config();

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
    process.exit(1);
});

console.log('--- Render Worker Initialization ---');
console.log('CWD:', process.cwd());
console.log('Node:', process.version);

const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const CALLBACK_URL = process.env.CALLBACK_URL;
const CALLBACK_TOKEN = process.env.CALLBACK_TOKEN;

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function reportSync(jobId: string, event: string, data: any, userId?: string) {
    try {
        await fetch(CALLBACK_URL!, {
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

const worker = new Worker('render-jobs', async (job: Job) => {
    const { jobId, analysisAssetId, userId } = job.data;
    console.log(`[${jobId}] Starting render for analysis ${analysisAssetId} for user ${userId}`);

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'remotion-'));
    const outputPath = path.join(tmpDir, 'output.mp4');

    try {
        await reportSync(jobId, 'stage', { status: 'compositing', progress: 10 }, userId);

        // 1. Download Analysis JSON
        const { data: assetData } = await supabase.from('job_assets').select('*').eq('id', analysisAssetId).single();
        if (!assetData) throw new Error('Analysis asset not found');

        const { data: fileData, error: downloadError } = await supabase.storage.from('assets').download(assetData.path);
        if (downloadError || !fileData) {
            console.error(`[${jobId}] Download failed for ${assetData.path}:`, downloadError);
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
            styleId: analysisJson.styleId, // Pass the resolved style ID
            fontFamily: analysisJson.fontFamily,
            animationEffect: analysisJson.animationEffect,
            lyricColor: analysisJson.lyricColor,
            lyricOpacity: analysisJson.lyricOpacity
        };

        // 2. Render Video
        await reportSync(jobId, 'stage', { status: 'compositing', progress: 30 });

        const compositionId = 'LyricVideo';
        const entry = path.join(process.cwd(), '../../apps/render/src/index.ts');

        console.log(`[${jobId}] Bundling...`);
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
                console.log(`[${jobId}] Progress: ${p}%`);
                reportSync(jobId, 'stage', { status: 'encoding', progress: p }, userId);
            }
        });

        // 3. Upload Output
        await reportSync(jobId, 'stage', { status: 'encoding', progress: 90 }, userId);
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

        await reportSync(jobId, 'stage', { status: 'completed', progress: 100 });
        console.log(`[${jobId}] Render completed!`);

    } catch (err: any) {
        console.error(`[${jobId}] Render failed:`, err);
        await reportSync(jobId, 'error', {
            message: err.message,
            errorCode: 'RENDER_FAILURE'
        });
        throw err;
    } finally {
        fs.rmSync(tmpDir, { recursive: true, force: true });
    }
}, {
    connection: {
        host: '127.0.0.1',
        port: 6379
    }
});

console.log('Render worker listening on Redis: 127.0.0.1:6379');
