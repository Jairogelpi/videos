
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';

dotenv.config({ path: path.join(__dirname, '../apps/api/.env') });

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function main() {
    console.log('Fetching recent job_assets...');
    const { data: jobAssets, error } = await supabase
        .from('job_assets')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(5);

    if (error) {
        console.error('Error fetching job_assets:', error);
        return;
    }

    // Log structure directly to console to see columns
    if (jobAssets && jobAssets.length > 0) {
        console.log('Sample job_asset keys:', Object.keys(jobAssets[0]));
        console.log('Sample job_asset:', jobAssets[0]);
    } else {
        console.log('No job_assets found.');
    }

    console.log('Fetching storage listing for "assets" bucket...');
    const { data: files, error: listError } = await supabase.storage.from('assets').list();

    // List contents of the specific failed job folder
    const failedJobId = 'a30e0908-2537-48c6-a6d8-97413f0f48d1';
    console.log(`Checking contents of folder: ${failedJobId}`);
    const { data: folderFiles } = await supabase.storage.from('assets').list(failedJobId);
    folderContents = folderFiles;

    const output = {
        jobAssets,
        bucketRoot: files,
        folderContents
    };

    fs.writeFileSync('debug_assets.json', JSON.stringify(output, null, 2));
    console.log('Dumped to debug_assets.json');
}

main();
