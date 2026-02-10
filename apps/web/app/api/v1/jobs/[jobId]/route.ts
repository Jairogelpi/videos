import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function GET(
    request: NextRequest,
    { params }: { params: { jobId: string } }
) {
    const { jobId } = params;

    const { data: job, error } = await supabase
        .from('jobs')
        .select('*, input_audio_asset_id(*), output_video_asset_id(*)')
        .eq('id', jobId)
        .single();

    if (error) {
        return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const getPublicUrl = (bucket: string, path: string | null) => {
        if (!path) return null;
        if (path.startsWith('http')) return path;
        return `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/${bucket}/${path}`;
    };

    return NextResponse.json({
        jobId: job.id,
        status: job.status,
        progress: job.progress,
        metrics: job.metadata?.metrics, // No logic asset in this MVP yet
        metadata: job.metadata,
        assets: {
            input_audio_url: getPublicUrl('assets', job.input_audio_asset_id?.path),
            output_video_url: getPublicUrl('assets', job.output_video_asset_id?.path)
        }
    });
}
