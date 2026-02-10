import { NextResponse } from 'next/server';
import { VISUAL_PRESETS } from '@repo/shared';

export async function GET() {
    return NextResponse.json({ styles: VISUAL_PRESETS });
}
