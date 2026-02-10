import Fastify from 'fastify';
import cors from '@fastify/cors';
import { createClient } from '@supabase/supabase-js';
import { WorkerCallbackSchema } from '@repo/shared';
import * as dotenv from 'dotenv';
import { Queue } from 'bullmq';
import { EventEmitter } from 'events';

dotenv.config();

const jobEvents = new EventEmitter();
// Increase limit for concurrent job listeners
jobEvents.setMaxListeners(100);

const jobQueue = new Queue('video-jobs', {
    connection: {
        host: process.env.REDIS_HOST || '127.0.0.1',
        port: parseInt(process.env.REDIS_PORT || '6379')
    }
});

const renderQueue = new Queue('render-jobs', {
    connection: {
        host: process.env.REDIS_HOST || '127.0.0.1',
        port: parseInt(process.env.REDIS_PORT || '6379')
    }
});

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

const fastify = Fastify({
    logger: true
});

fastify.register(cors);

// 1. Health check
fastify.get('/health', async () => {
    return { status: 'ok' };
});

// 2. Sign Upload URL (Supabase Storage)
fastify.post('/v1/uploads/sign', async (request, reply) => {
    const { fileName, contentType, userId: bodyUserId } = request.body as { fileName: string, contentType: string, userId?: string };
    const userId = bodyUserId || '00000000-0000-0000-0000-000000000000';

    const sanitizedFileName = fileName.replace(/[^a-zA-Z0-9.-]/g, '_');
    const path = `${userId}/${Date.now()}-${sanitizedFileName}`;

    const { data, error } = await supabase.storage
        .from('assets')
        .createSignedUploadUrl(path);

    if (error) {
        return reply.status(500).send({ error: error.message });
    }

    return {
        uploadUrl: data.signedUrl,
        assetPath: path,
        publicUrl: `${process.env.SUPABASE_URL}/storage/v1/object/public/assets/${path}`
    };
});

import { VISUAL_PRESETS } from '@repo/shared';

// 2a. Get Visual Library
fastify.get('/v1/styles', async () => {
    return { styles: VISUAL_PRESETS };
});

// 2b. AI Style Selector (Prompt to Style)
fastify.post('/v1/styles/determine', async (request, reply) => {
    const { prompt } = request.body as { prompt: string };
    if (!prompt) return reply.status(400).send({ error: 'Prompt is required' });

    console.log(`[AI] Determining style for prompt: "${prompt}"`);

    // Semantic Matching Logic (V1 Universal Processor)
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

    return {
        bestStyleId: bestStyle.id,
        config: bestStyle,
        aiReasoning: `Matched your prompt "${prompt}" with ${bestStyle.name} based on semantic alignment.`
    };
});

// 3. Create Job
fastify.post('/v1/jobs', async (request, reply) => {
    const { title, inputAudioAssetId, projectId, userId: bodyUserId, mood, styleId: bodyStyleId, prompt, position, startTime, endTime, targetLanguage, audioUrl, fontSize, fontFamily, animationEffect, lyricColor, lyricOpacity } = request.body as {
        title: string,
        inputAudioAssetId: string,
        projectId?: string,
        userId?: string,
        mood?: string,
        styleId?: string,
        prompt?: string,
        position?: 'top' | 'center' | 'bottom',
        startTime?: number,
        endTime?: number,
        targetLanguage?: string,
        audioUrl?: string,
        fontSize?: number,
        fontFamily?: string,
        animationEffect?: string,
        lyricColor?: string,
        lyricOpacity?: number
    };

    const userId = bodyUserId || '00000000-0000-0000-0000-000000000000';

    // AI STYLE SELECTION: Resolve Prompt to Style if provided
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
            metadata: { mood, styleId, position, startTime, endTime, targetLanguage, fontSize, prompt, fontFamily, animationEffect, lyricColor, lyricOpacity } // Store in metadata column
        })
        .select()
        .single();

    if (error) {
        console.error('[JobOrchestrator] Supabase Insert Error:', error);
        return reply.status(500).send({ error: error.message });
    }

    // 2. Hybrid Cloud Dispatch (Modal vs Local)
    const modalUrl = process.env.MODAL_WEBHOOK_URL;

    if (modalUrl) {
        console.log(`[JobOrchestrator] ☁️ Hybrid Cloud Mode: Offloading job ${job.id} to Modal...`);
        try {
            const modalPayload = {
                job_id: job.id,
                asset_id: resolvedAudioId,
                user_id: userId,
                prompt: prompt || mood || "Creative visualization",
                style: styleId || "cinematic"
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

            // Job status remains 'queued' but is now managed by cloud
            return { jobId: job.id, status: 'queued', mode: 'cloud' };

        } catch (cloudError: any) {
            console.error(`[JobOrchestrator] ❌ Cloud Dispatch Failed: ${cloudError.message}`);
            console.log(`[JobOrchestrator] ⚠️ Falling back to LOCAL processing...`);
            // Fallthrough to local queue if cloud fails? 
            // For now, let's fallthrough.
        }
    }

    // 3. Local Enqueue (BullMQ) - Fallback or Default
    try {
        console.log(`[JobOrchestrator] 🏠 Local Mode: Enqueuing job ${job.id} to BullMQ...`);
        await jobQueue.add('process-video', {
            jobId: job.id,
            userId,
            inputAudioAssetId: resolvedAudioId,
            mood: mood || 'default',
            styleId: styleId || 'tiktok_bold',
            position: position || 'center',
            startTime: startTime ?? 0,
            endTime: endTime ?? 60,
            targetLanguage,
            fontSize,
            fontFamily,
            animationEffect,
            lyricColor,
            lyricOpacity,
            bgPrompt: prompt
        });
    } catch (queueError) {
        console.error('[JobOrchestrator] BullMQ Enqueue Error:', queueError);
        return reply.status(500).send({ error: 'Failed to enqueue job' });
    }

    return { jobId: job.id, status: job.status };
});

// 3b. List Jobs (History)
fastify.get('/v1/jobs', async (request, reply) => {
    const { userId: bodyUserId } = request.query as { userId?: string };
    const userId = bodyUserId || '00000000-0000-0000-0000-000000000000';

    const { data: jobs, error } = await supabase
        .from('jobs')
        .select('*, input_audio_asset_id(*), output_video_asset_id(*)')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

    if (error) {
        return reply.status(500).send({ error: error.message });
    }

    return { jobs };
});

// 4. Job Status
fastify.get('/v1/jobs/:jobId', async (request, reply) => {
    const { jobId } = request.params as { jobId: string };

    const { data: job, error } = await supabase
        .from('jobs')
        .select('*, input_audio_asset_id(*), output_video_asset_id(*)')
        .eq('id', jobId)
        .single();

    if (error) {
        return reply.status(404).send({ error: 'Job not found' });
    }

    const getPublicUrl = (bucket: string, path: string | null) => {
        if (!path) return null;
        if (path.startsWith('http')) return path;
        return `${process.env.SUPABASE_URL}/storage/v1/object/public/${bucket}/${path}`;
    };

    return {
        jobId: job.id,
        status: job.status,
        progress: job.progress,
        metrics: job.metadata?.metrics || job.analysis_json_asset_id?.metadata?.metrics,
        metadata: job.metadata,
        assets: {
            input_audio_url: getPublicUrl('assets', job.input_audio_asset_id?.path),
            output_video_url: getPublicUrl('outputs', job.output_video_asset_id?.path)
        }
    };
});

// 5. Real-time Events (SSE)
fastify.get('/v1/jobs/:jobId/events', (request, reply) => {
    const { jobId } = request.params as { jobId: string };

    reply.raw.setHeader('Content-Type', 'text/event-stream');
    reply.raw.setHeader('Cache-Control', 'no-cache');
    reply.raw.setHeader('Connection', 'keep-alive');

    const onEvent = (eventData: any) => {
        if (eventData.jobId === jobId) {
            reply.raw.write(`data: ${JSON.stringify(eventData)}\n\n`);
        }
    };

    jobEvents.on('update', onEvent);

    const interval = setInterval(() => {
        reply.raw.write(`data: ${JSON.stringify({ type: 'keep-alive' })}\n\n`);
    }, 15000);

    request.raw.on('close', () => {
        jobEvents.off('update', onEvent);
        clearInterval(interval);
    });
});

// 6. Worker Callback (Internal)
fastify.post('/v1/internal/workers/callback', async (request, reply) => {
    const result = WorkerCallbackSchema.safeParse(request.body);
    if (!result.success) {
        return reply.status(400).send({ error: 'Invalid callback payload', details: result.error });
    }

    const payload = result.data;
    const { jobId } = payload;
    fastify.log.info({ workerEvent: payload }, 'Received worker event');

    // 1. Database Updates (Supabase)
    const { data: job } = await supabase.from('jobs').select('*').eq('id', jobId).single();

    if (payload.event === 'stage') {
        const { error } = await supabase
            .from('jobs')
            .update({
                status: payload.status,
                progress: payload.progress,
                // Optional: Store metrics in job metadata if provided
                metadata: payload.metrics ? { ...(job?.metadata || {}), metrics: payload.metrics } : undefined
            })
            .eq('id', jobId);

        if (error) fastify.log.error({ error }, 'Failed to update job status');
    }

    if (payload.event === 'asset') {
        // Create asset record
        const { data: asset, error: assetError } = await supabase
            .from('job_assets')
            .insert({
                user_id: payload.userId || '00000000-0000-0000-0000-000000000000',
                job_id: jobId,
                bucket: payload.kind === 'final_video' ? 'outputs' : 'assets',
                path: payload.url,
                kind: payload.kind
            })
            .select()
            .single();

        if (assetError) {
            fastify.log.error({ assetError }, 'Failed to create asset');
        } else {
            // Update job pointers
            if (payload.kind === 'final_video') {
                await supabase.from('jobs').update({ output_video_asset_id: asset.id }).eq('id', jobId);
            } else if (payload.kind === 'analysis_json') {
                await supabase.from('jobs').update({ analysis_json_asset_id: asset.id }).eq('id', jobId);

                // ORCHESTRATION: Trigger Render Stage
                fastify.log.info({ jobId }, 'Triggering render job');
                await renderQueue.add('render-video', {
                    jobId,
                    userId: '00000000-0000-0000-0000-000000000000',
                    analysisAssetId: asset.id
                });
            }
        }
    }

    // 2. SSE Broadcast
    jobEvents.emit('update', {
        jobId,
        type: 'progress',
        status: payload.event === 'stage' ? payload.status : undefined,
        progress: payload.event === 'stage' ? payload.progress : undefined,
        metrics: payload.event === 'stage' ? payload.metrics : undefined,
        message: payload.event === 'stage' ? `Executing ${payload.status}` :
            payload.event === 'asset' ? `Asset generated: ${payload.kind}` :
                payload.event === 'error' ? `Error: ${payload.message}` : 'Status updated',
        assetUrl: payload.event === 'asset' ? payload.url : undefined
    });

    // 3. Persistent Event Log
    await supabase.from('job_events').insert({
        job_id: jobId,
        event_type: payload.event,
        payload: payload
    });

    return { success: true };
});

const start = async () => {
    try {
        await fastify.listen({ port: 3001, host: '0.0.0.0' });
    } catch (err) {
        fastify.log.error(err);
        process.exit(1);
    }
};

start();
