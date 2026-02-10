import { NextRequest, NextResponse } from 'next/server';
import { VISUAL_PRESETS } from '@repo/shared';

export async function POST(request: NextRequest) {
    try {
        const { prompt } = await request.json();
        if (!prompt) return NextResponse.json({ error: 'Prompt is required' }, { status: 400 });

        console.log(`[AI] Determining style for prompt: "${prompt}"`);

        // Semantic Matching Logic (V1 Universal Processor)
        const words = prompt.toLowerCase().split(/\s+/);
        let bestStyle = VISUAL_PRESETS[0];
        let maxScore = -1;

        for (const style of VISUAL_PRESETS) {
            let score = 0;
            for (const word of words) {
                if (style.tags.includes(word)) score += 2;
                if (style.name.toLowerCase().includes(word)) score += 1;
            }
            if (score > maxScore) {
                maxScore = score;
                bestStyle = style;
            }
        }

        return NextResponse.json({
            bestStyleId: bestStyle.id,
            config: bestStyle,
            aiReasoning: `Matched your prompt "${prompt}" with ${bestStyle.name} based on semantic alignment.`
        });
    } catch (err: any) {
        console.error('[API] Style Determine Error:', err);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
