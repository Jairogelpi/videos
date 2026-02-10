
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(__dirname, '../apps/api/.env') });

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

const jobId = process.argv[2];

if (!jobId) {
    console.error('Please provide a job ID');
    process.exit(1);
}

async function checkJob() {
    console.log(`Checking job ${jobId}...`);
    const { data: job, error } = await supabase
        .from('jobs')
        .select('*, input_audio_asset_id(*), output_video_asset_id(*)')
        .eq('id', jobId)
        .single();

    if (error) {
        console.error('Error fetching job:', error);
        return;
    }

    console.log('Job:', job);
}

checkJob();
