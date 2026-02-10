
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.join(__dirname, '../apps/api/.env') });

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function main() {
    console.log('Fetching latest job...');
    const { data: jobs, error } = await supabase
        .from('jobs')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(1);

    if (error) {
        console.error('Error fetching job:', error);
        return;
    }

    if (!jobs || jobs.length === 0) {
        console.log('No jobs found.');
        return;
    }

    const job = jobs[0];
    console.log('--- Latest Job ---');
    console.log(`ID: ${job.id}`);
    console.log(`User ID: ${job.user_id}`);
    console.log(`Status: ${job.status}`);
    console.log(`Input Audio Asset ID: ${job.input_audio_asset_id}`);
    console.log(`Analysis JSON Asset ID: ${job.analysis_json_asset_id}`);
    console.log(`Output Video Asset ID: ${job.output_video_asset_id}`);
    console.log(`Created At: ${job.created_at}`);
    console.log(`Metadata:`, JSON.stringify(job.metadata, null, 2));

    // Manually fetch recent assets (last 50) and filter in JS
    console.log('\nFetching recent assets (unfiltered)...');
    const { data: recentAssets, error: assetsError } = await supabase
        .from('job_assets')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50);

    if (assetsError) {
        console.error('Error fetching assets:', assetsError);
    } else {
        console.log('\n--- Related Assets ---');
        // Filter for assets related to this job (by job_id OR by approximate timestamp/user if job_id missing for input)
        const jobDate = new Date(job.created_at);

        recentAssets?.forEach((asset: any) => {
            const assetDate = new Date(asset.created_at);
            const isAfterJob = assetDate >= jobDate; // Roughly
            const matchesJobId = asset.job_id === job.id;
            const matchesUser = asset.user_id === job.user_id;

            // We want assets that:
            // 1. Have this job_id
            // 2. OR are input_audio for this user created recently
            // 3. OR are linked via IDs in the job record

            const isLinked =
                matchesJobId ||
                asset.id === job.input_audio_asset_id ||
                asset.id === job.analysis_json_asset_id ||
                asset.id === job.output_video_asset_id;

            if (isLinked) {
                console.log(`[${asset.kind}] ${asset.bucket}/${asset.path}`);
                console.log(`    ID: ${asset.id}`);
                console.log(`    JobId: ${asset.job_id}`);
                console.log(`    Created: ${asset.created_at}`);
            }
        });
    }

    // Manually fetch events
    console.log('\nFetching events...');
    const { data: jobEvents, error: eventsError } = await supabase
        .from('job_events')
        .select('*')
        .eq('job_id', job.id)
        .order('created_at', { ascending: false });

    if (eventsError) {
        console.error('Error fetching events:', eventsError);
    } else {
        console.log('\n--- Recent Events ---');
        jobEvents?.forEach((event: any) => {
            console.log(`[${event.created_at}] ${event.event_type}:`);
            console.log(JSON.stringify(event.payload, null, 2));
        });
    }

    // Check availability of output video
    if (job.output_video_asset_id) {
        console.log('\nChecking output video file availability...');
        // Find asset in recent list or fetch it
        let asset = recentAssets?.find((a: any) => a.id === job.output_video_asset_id);

        if (!asset) {
            const { data: specificAsset } = await supabase.from('job_assets').select('*').eq('id', job.output_video_asset_id).single();
            asset = specificAsset;
        }

        if (asset) {
            const { data, error: dlError } = await supabase.storage
                .from(asset.bucket)
                .download(asset.path);
            if (dlError) console.error('Download check failed:', dlError);
            else console.log(`Download check passed. Size: ${data.size} bytes`);
        } else {
            console.error(`Output video asset ID ${job.output_video_asset_id} set, but record not found in DB!`);
        }
    } else {
        console.log('\nNo Output Video Asset ID assigned to job yet.');
    }
}

main();
