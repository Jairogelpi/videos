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
            animationEffect, lyricColor, lyricOpacity,
            videoTitle, titleFontFamily
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

        // AI STYLE SELECTION: Only auto-resolve if NO style was explicitly chosen
        let styleId = bodyStyleId;
        if (!styleId && prompt) {
            console.log(`[JobOrchestrator] No style selected. Auto-resolving from prompt: "${prompt}"`);
            // styleId is already undefined if not provided. Let it be passed as such to the worker.
            // WORKER LOGIC handles undefined style by using the AI Director (Gemini) to determine style from context.
            console.log(`[JobOrchestrator] No style selected. Passing 'Auto' (undefined) to worker.`);
        } else if (styleId) {
            console.log(`[JobOrchestrator] Using user-selected style: ${styleId}`);
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
                    animationEffect, lyricColor, lyricOpacity,
                    videoTitle, titleFontFamily
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
                const origin = process.env.PUBLIC_CALLBACK_URL || request.nextUrl.origin;
                const modalPayload = {
                    job_id: job.id,
                    asset_id: resolvedAudioId,
                    user_id: userId,
                    prompt: prompt || mood || "Creative visualization",
                    style: styleId, // undefined if Auto
                    fontFamily,
                    animationEffect,
                    lyricOpacity,
                    position,
                    fontSize,
                    targetLanguage,
                    video_title: videoTitle,
                    title_font_family: titleFontFamily,
                    lyricColor,
                    resolution: job.resolution || '1080p',
                    fps: job.fps || 30,
                    duration_sec: job.duration_sec || 60,
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
