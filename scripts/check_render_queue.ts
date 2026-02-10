
import { Queue } from 'bullmq';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.join(__dirname, '../apps/api/.env') });

const REDIS_HOST = process.env.REDIS_HOST || '127.0.0.1';
const REDIS_PORT = parseInt(process.env.REDIS_PORT || '6379');

const renderQueue = new Queue('render-jobs', {
    connection: {
        host: REDIS_HOST,
        port: REDIS_PORT
    }
});

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function main() {
    console.log(`Checking job 7 in render-jobs...`);
    const job = await renderQueue.getJob('7');

    if (!job) {
        console.log('Job 7 not found.');
        return;
    }

    const { analysisAssetId } = job.data;
    console.log(`Job Data AnalysisAssetId: "${analysisAssetId}"`);

    if (analysisAssetId) {
        console.log(`\nQuerying job_assets for ID: ${analysisAssetId}...`);
        const { data: asset, error } = await supabase
            .from('job_assets')
            .select('*')
            .eq('id', analysisAssetId)
            .single();

        if (error) {
            console.error('Error fetching asset:', error);
        } else {
            console.log('Asset found in DB:');
            console.log(`Bucket: "${asset.bucket}"`);
            console.log(`Path: "${asset.path}"`);

            // Test 1: Download using DB path
            console.log(`\n[Test 1] Downloading from "${asset.bucket}" with path "${asset.path}"...`);
            const { data: fileData, error: downloadError } = await supabase.storage
                .from(asset.bucket)
                .download(asset.path);

            if (downloadError) {
                console.error('[Test 1] FAILED:', downloadError);
            } else {
                console.log(`[Test 1] SUCCESS! Size: ${fileData?.size} bytes`);
            }

            // Test 2: Download using hardcoded known path
            const knownPath = 'a30e0908-2537-48c6-a6d8-97413f0f48d1/analysis.json';
            console.log(`\n[Test 2] Downloading from "assets" with known path "${knownPath}"...`);
            const { data: fileData2, error: downloadError2 } = await supabase.storage
                .from('assets')
                .download(knownPath);

            if (downloadError2) {
                console.error('[Test 2] FAILED:', downloadError2);
            } else {
                console.log(`[Test 2] SUCCESS! Size: ${fileData2?.size} bytes`);
            }
        }
    }

    process.exit(0);
}

main();
