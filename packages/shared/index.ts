import { z } from 'zod';
export * from './visual_config';

// Job Stages for V1 Premium
export const JobStatusSchema = z.enum([
    'queued',
    'preprocessing',
    'transcribing',
    'aligning',
    'hook_selecting',
    'compositing',
    'encoding',
    'completed',
    'failed',
    'canceled'
]);

export const AssetKindSchema = z.enum([
    'input_audio',
    'vocal_stem',
    'cleaned_vocals',
    'analysis_json',
    'hook_audio',
    'draft_video',
    'final_video'
]);

// analysis.json Contract
export const WordSchema = z.object({
    t0: z.number(),
    t1: z.number(),
    w: z.string(),
    conf: z.number(),
    lang: z.string().optional(),
    tw: z.string().optional(), // Translated word/text
    flags: z.array(z.string()).optional(),
    type: z.string().default('lyric')
});

export const AnalysisJsonSchema = z.object({
    jobId: z.string().uuid(),
    hook: z.object({
        start: z.number(),
        end: z.number(),
        score: z.number()
    }),
    words: z.array(WordSchema),
    languageSpans: z.array(z.object({
        t0: z.number(),
        t1: z.number(),
        lang: z.string()
    })),
    metrics: z.object({
        alignScore: z.number(),
        coverage: z.number(),
        uncertainRate: z.number(),
        sourceAgreementRate: z.number(),
        werProxy: z.number().optional()
    }),
    alignment: z.object({
        method: z.string(),
        alignScore: z.number(),
        alignedRatio: z.number(),
        fallbackUsed: z.boolean(),
        granularity: z.enum(['word', 'phrase', 'caption']).default('word')
    }).optional(),
    profiling: z.object({
        snr: z.number(),
        reverbLevel: z.number(),
        tempo: z.number()
    }).optional()
});

// Worker Callback Payload
export const WorkerCallbackSchema = z.discriminatedUnion('event', [
    z.object({
        event: z.literal('stage'),
        jobId: z.string().uuid(),
        userId: z.string().uuid().optional(),
        status: JobStatusSchema,
        progress: z.number().min(0).max(100),
        metrics: z.object({
            snr: z.number().optional(),
            reverbLevel: z.number().optional(),
            tempo: z.number().optional(),
            alignScore: z.number().optional()
        }).optional()
    }),
    z.object({
        event: z.literal('asset'),
        jobId: z.string().uuid(),
        userId: z.string().uuid().optional(),
        kind: AssetKindSchema,
        url: z.string(),
        metadata: z.record(z.any()).optional()
    }),
    z.object({
        event: z.literal('error'),
        jobId: z.string().uuid(),
        userId: z.string().uuid().optional(),
        errorCode: z.string(),
        message: z.string(),
        retryable: z.boolean().default(false)
    })
]);
