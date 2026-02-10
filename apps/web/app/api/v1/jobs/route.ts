import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import { createClient } from '@/lib/supabase-server';
import { VISUAL_PRESETS } from '@repo/shared';

// 1. GET /v1/jobs
export async function GET(request: NextRequest) {
    // Use server client to get the real user
    const serverSupabase = await createClient();
    const { data: { user } } = await serverSupabase.auth.getUser();

    if (!user) {
        return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { data: jobs, error } = await supabase
        .from('jobs')
        .select('*, input_audio_asset_id(*), output_video_asset_id(*)')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

    if (error) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({ jobs });
}

// 2. POST /v1/jobs
export async function POST(request: NextRequest) {
    try {
        // Get real authenticated user
        const serverSupabase = await createClient();
        const { data: { user } } = await serverSupabase.auth.getUser();

        if (!user) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const userId = user.id;

        const body = await request.json();
        const {
            title, inputAudioAssetId, projectId: bodyProjectId, mood,
            styleId: bodyStyleId, prompt, position, startTime, endTime,
            targetLanguage, audioUrl, fontSize, fontFamily,
            animationEffect, lyricColor, lyricOpacity
        } = body;

        // Resolve project: use provided or find user's default project
        let projectId = bodyProjectId;
        if (!projectId) {
            const { data: defaultProject } = await supabase
                .from('projects')
                .select('id')
                .eq('user_id', userId)
                .order('created_at', { ascending: true })
                .limit(1)
                .single();
            projectId = defaultProject?.id || null;
        }

        // AI STYLE SELECTION: Resolve Prompt to Style if provided (Mirroring Fastify logic)
        let styleId = bodyStyleId;
        if (prompt) {
            console.log(`[JobOrchestrator] Resolving prompt: "${prompt}"`);
            const words = prompt.toLowerCase().split(/\s+/);
            let bestStyle = VISUAL_PRESETS[0];
            let maxScore = -1;
            for (const style of VISUAL_PRESETS) {
                let score = 0;
                for (const word of words) {
                    if (style.tags.includes(word)) score += 2;
                    if (style.name.toLowerCase().includes(word)) score += 1;
                }
                if (score > maxScore) {
                    maxScore = score;
                    bestStyle = style;
                }
            }
            styleId = bestStyle.id;
            console.log(`[JobOrchestrator] Resolved to style: ${styleId}`);
        }

        // 0. CREATE ASSET RECORD if audioUrl is provided but no ID
        let resolvedAudioId = inputAudioAssetId;
        if (!resolvedAudioId && audioUrl) {
            console.log(`[JobOrchestrator] Creating dynamic asset for: ${audioUrl}`);
            const { data: asset, error: assetError } = await supabase
                .from('job_assets')
                .insert({
                    user_id: userId,
                    bucket: 'assets',
                    path: audioUrl,
                    kind: 'input_audio'
                })
                .select()
                .single();

            if (!assetError && asset) {
                resolvedAudioId = asset.id;
            } else {
                console.error('[JobOrchestrator] Failed to create dynamic asset:', assetError);
            }
        }

        // 1. Persist in Supabase
        const { data: job, error } = await supabase
            .from('jobs')
            .insert({
                user_id: userId,
                project_id: projectId,
                input_audio_asset_id: resolvedAudioId,
                status: 'queued',
                metadata: {
                    mood, styleId, position, startTime, endTime,
                    targetLanguage, fontSize, prompt, fontFamily,
                    animationEffect, lyricColor, lyricOpacity
                }
            })
            .select()
            .single();

        if (error) {
            console.error('[JobOrchestrator] Supabase Insert Error:', error);
            return NextResponse.json({ error: error.message }, { status: 500 });
        }

        // 2. Hybrid Cloud Dispatch (Modal Handoff)
        const modalUrl = process.env.MODAL_WEBHOOK_URL;

        if (modalUrl) {
            console.log(`[JobOrchestrator] ☁️ Hybrid Cloud Mode: Offloading job ${job.id} to Modal...`);
            try {
                const origin = request.nextUrl.origin;
                const modalPayload = {
                    job_id: job.id,
                    asset_id: resolvedAudioId, // We might need to resolve the path if Modal expects it
                    user_id: userId,
                    prompt: prompt || mood || "Creative visualization",
                    style: styleId || "cinematic",
                    callback_url: `${origin}/api/v1/internal/workers/callback`
                };

                const modalRes = await fetch(modalUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(modalPayload)
                });

                if (!modalRes.ok) {
                    const errText = await modalRes.text();
                    throw new Error(`Modal Webhook Failed: ${modalRes.status} ${errText}`);
                }

                const modalData = await modalRes.json();
                console.log(`[JobOrchestrator] ✅ Modal Accepted:`, modalData);

                return NextResponse.json({ jobId: job.id, status: 'queued', mode: 'cloud' });

            } catch (cloudError: any) {
                console.error(`[JobOrchestrator] ❌ Cloud Dispatch Failed: ${cloudError.message}`);
                // In Cloud-First mode, we might not want local fallback to be automatic
                // but for developer UX we could keep it.
            }
        }

        // 3. Fallback: If no Modal URL or Cloud Dispatch Failed, we don't do local BullMQ here
        // as Vercel functions cannot easily talk to a local Redis. 
        // Instead, we just return the job ID. The user can run a local worker separately.

        return NextResponse.json({ jobId: job.id, status: 'queued', message: 'Job created. Waiting for worker.' });

    } catch (err: any) {
        console.error('[API] Jobs Create Error:', err);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
