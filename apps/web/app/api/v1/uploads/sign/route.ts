import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

export async function POST(request: NextRequest) {
    try {
        const { fileName, contentType, userId: bodyUserId } = await request.json();
        const userId = bodyUserId || '00000000-0000-0000-0000-000000000000';

        const sanitizedFileName = fileName.replace(/[^a-zA-Z0-9.-]/g, '_');
        const path = `${userId}/${Date.now()}-${sanitizedFileName}`;

        const { data, error } = await supabase.storage
            .from('assets')
            .createSignedUploadUrl(path);

        if (error) {
            console.error('[API] Supabase Signed URL Error:', error);
            return NextResponse.json({ error: error.message }, { status: 500 });
        }

        return NextResponse.json({
            uploadUrl: data.signedUrl,
            assetPath: path,
            publicUrl: `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/assets/${path}`
        });
    } catch (err: any) {
        console.error('[API] Sign Upload Error:', err);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
