
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.join(__dirname, '../apps/api/.env') });

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function main() {
    // We need a valid Job ID. Let's fetch the most recent job.
    const { data: jobs } = await supabase
        .from('jobs')
        .select('id, user_id')
        .order('created_at', { ascending: false })
        .limit(1);

    if (!jobs || jobs.length === 0) {
        console.error('No jobs found to test with.');
        return;
    }

    const job = jobs[0];
    console.log('Testing with Job ID:', job.id);

    const payload = {
        jobId: job.id,
        userId: job.user_id,
        kind: 'analysis_json',
        url: `${job.id}/test_analysis.json`
    };

    console.log('Attempting to insert asset with:', payload);

    const { data: asset, error: assetError } = await supabase
        .from('job_assets')
        .insert({
            user_id: payload.userId,
            job_id: payload.jobId,
            bucket: 'assets',
            path: payload.url,
            kind: payload.kind
        })
        .select()
        .single();

    if (assetError) {
        console.error('FAILED to create asset:', assetError);
    } else {
        console.log('SUCCESS: Asset created:', asset);
        // Clean up
        await supabase.from('job_assets').delete().eq('id', asset.id);
        console.log('Cleaned up test asset.');
    }
}

main();
