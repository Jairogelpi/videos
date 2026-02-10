
import { createClient } from '@supabase/supabase-js'
import fs from 'fs'
import path from 'path'

// Manually load env vars from .env.local because dotenv might not be available
const envPath = path.resolve(process.cwd(), '.env.local');
console.log(`Loading env from ${envPath}`);

if (fs.existsSync(envPath)) {
    const envConfig = fs.readFileSync(envPath, 'utf8');
    envConfig.split('\n').forEach(line => {
        const [key, value] = line.split('=');
        if (key && value) {
            process.env[key.trim()] = value.trim();
        }
    });
}
else {
    console.warn('.env.local not found!');
}

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

if (!supabaseUrl || !supabaseServiceKey) {
    console.error('Missing Supabase credentials in .env.local');
    process.exit(1);
}

console.log(`Connecting to ${supabaseUrl}...`)
const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function checkBuckets() {
    const { data, error } = await supabase.storage.listBuckets()
    if (error) {
        console.error('Error listing buckets:', error)
        return
    }
    console.log('Buckets:', data?.map(b => b.name))

    // Check for 'assets' bucket
    const assetsExists = data?.some(b => b.name === 'assets')
    if (!assetsExists) {
        console.log('Creating "assets" bucket...')
        const { error: createError } = await supabase.storage.createBucket('assets', {
            public: true,
            fileSizeLimit: 52428800, // 50MB
            allowedMimeTypes: ['audio/*']
        })
        if (createError) console.error('Error creating assets bucket:', createError)
        else console.log('Assets bucket created successfully.')
    } else {
        console.log('"assets" bucket already exists.')
    }

    // Check for 'outputs' bucket (for render results)
    const outputsExists = data?.some(b => b.name === 'outputs')
    if (!outputsExists) {
        console.log('Creating "outputs" bucket...')
        const { error: createOutputError } = await supabase.storage.createBucket('outputs', {
            public: true,
            fileSizeLimit: 104857600, // 100MB
            allowedMimeTypes: ['video/*']
        })
        if (createOutputError) console.error('Error creating outputs bucket:', createOutputError)
        else console.log('Outputs bucket created successfully.')
    } else {
        console.log('"outputs" bucket already exists.')
    }
}

checkBuckets()
