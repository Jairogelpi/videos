
import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

// Header to check for security
const CALLBACK_TOKEN = process.env.CALLBACK_TOKEN!;

export async function POST(request: NextRequest) {
    try {
        // 1. Security Check
        const authHeader = request.headers.get('authorization');
        if (!authHeader || authHeader !== `Bearer ${CALLBACK_TOKEN}`) {
            console.error('[Callback] Unauthorized access attempt');
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const body = await request.json();
        const { jobId, event, ...data } = body;

        console.log(`[Callback] Job ${jobId} Event: ${event}`, data);

        if (!jobId) {
            return NextResponse.json({ error: 'Missing jobId' }, { status: 400 });
        }

        // 2. Handle Events
        if (event === 'stage') {
            const { status, progress, metrics } = data;

            const updateData: any = {
                status: status === 'completed' ? 'completed' : 'processing', // Map worker status to DB enum if needed
                progress
            };

            // Map worker 'completed' to DB 'completed'? 
            // The worker says "completed" when render finishes.
            if (status === 'completed') {
                updateData.status = 'completed';
            }

            // Should we update metadata with metrics?
            if (metrics) {
                // Fetch current metadata to merge? Or Supabase jsonb path update?
                // Supabase JS doesn't support deep merge easily without fetching first or using RPC.
                // Let's fetch first.
                const { data: job } = await supabase.from('jobs').select('metadata').eq('id', jobId).single();
                if (job) {
                    updateData.metadata = {
                        ...job.metadata,
                        ...metrics // Merge new metrics
                    };
                }
            }

            await supabase.from('jobs').update(updateData).eq('id', jobId);

        } else if (event === 'asset') {
            const { kind, url, metadata } = data;
            // Create or Update Asset Record?
            // The workers upload file then tell us. 
            // We usually insert into job_assets.

            // url is typically "userId/jobId_...".
            // bucket is 'assets' or 'outputs'.
            const bucket = kind === 'final_video' ? 'outputs' : 'assets';

            // Check if asset exists?
            // Usually worker uploads and we trust it.
            // We insert a new record in job_assets.

            const { data: asset, error } = await supabase.from('job_assets').insert({
                job_id: jobId,
                user_id: data.userId, // Worker should send userId
                kind: kind,
                path: url,
                bucket: bucket
            }).select().single();

            if (error) {
                console.error('[Callback] Asset Insert Error:', error);
                return NextResponse.json({ error: error.message }, { status: 500 });
            }

            // If it's the final video, update the job's output_video_asset_id
            if (kind === 'final_video') {
                await supabase.from('jobs').update({
                    output_video_asset_id: asset.id,
                    status: 'completed',
                    progress: 100
                }).eq('id', jobId);
            }

        } else if (event === 'error') {
            const { message, errorCode } = data;
            await supabase.from('jobs').update({
                status: 'failed',
                error_message: message
            }).eq('id', jobId);
        }

        return NextResponse.json({ success: true });

    } catch (err: any) {
        console.error('[Callback] Error:', err);
        return NextResponse.json({ error: err.message }, { status: 500 });
    }
}
