'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import {
    CheckCircle2,
    Clock,
    Activity,
    Scissors,
    Type,
    Music,
    Video,
    AlertCircle,
    ChevronRight,
    ShieldCheck
} from 'lucide-react';
import { cn } from '@/lib/utils';

type JobStatus = 'queued' | 'preprocessing' | 'transcribing' | 'aligning' | 'hook_selecting' | 'compositing' | 'encoding' | 'completed' | 'failed';

interface JobState {
    status: JobStatus;
    progress: number;
    stage: string;
    metrics?: {
        snr?: number;
        reverbLevel?: number;
        tempo?: number;
        alignScore?: number;
    };
    metadata?: {
        mood?: string;
        styleId?: string;
        targetLanguage?: string;
    };
    videoUrl?: string; // High-res final video
}

const STAGES: { status: JobStatus; label: string; icon: any }[] = [
    { status: 'queued', label: 'En Cola', icon: Clock },
    { status: 'preprocessing', label: 'Demucs Isolation', icon: Activity },
    { status: 'transcribing', label: 'Transcripción', icon: Type },
    { status: 'aligning', label: 'Sincronía 10ms', icon: Scissors },
    { status: 'hook_selecting', label: 'REP Engine', icon: Music },
    { status: 'compositing', label: 'Remotion HD', icon: Activity },
    { status: 'completed', label: 'Entregado', icon: CheckCircle2 },
];

export default function JobProgressPage() {
    const { jobId } = useParams();
    const [job, setJob] = useState<JobState>({
        status: 'queued',
        progress: 0,
        stage: 'Conectando con el orquestador...'
    });

    useEffect(() => {
        let interval: NodeJS.Timeout;

        // 1. Initial & Polling Fetch
        const fetchJob = async () => {
            try {
                const res = await fetch(`/api/v1/jobs/${jobId}`);
                if (res.ok) {
                    const data = await res.json();
                    setJob({
                        status: data.status,
                        progress: data.progress,
                        stage: `Estado: ${data.status}`,
                        metrics: data.metrics,
                        metadata: data.metadata,
                        videoUrl: data.assets?.output_video_url
                    });

                    // Stop polling if completed or failed
                    if (data.status === 'completed' || data.status === 'failed' || data.progress >= 100) {
                        clearInterval(interval);
                    }
                }
            } catch (err) {
                console.error('Failed to fetch job status', err);
            }
        };

        fetchJob();

        // High-frequency polling loop (1 second)
        interval = setInterval(fetchJob, 1000);

        // 2. Real-time Updates (SSE) - Complementary
        const eventSource = new EventSource(`/api/v1/jobs/${jobId}/events`);

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'progress') {
                setJob(prev => ({
                    ...prev,
                    status: data.status || prev.status,
                    progress: data.progress ?? prev.progress,
                    stage: data.message || prev.stage,
                    metrics: data.metrics || prev.metrics,
                    videoUrl: data.assetUrl || prev.videoUrl
                }));

                if (data.progress >= 100 || data.status === 'completed') {
                    clearInterval(interval);
                }
            }
        };

        eventSource.onerror = () => {
            console.error('SSE connection failed');
            eventSource.close();
        };

        return () => {
            clearInterval(interval);
            eventSource.close();
        };
    }, [jobId]);

    const currentStageIndex = STAGES.findIndex(s => s.status === job.status);

    return (
        <main className="min-h-screen bg-[#050505] text-white p-6 md:p-12 font-sans overflow-x-hidden">
            <div className="max-w-6xl mx-auto space-y-12 animate-in fade-in duration-1000">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 py-6 border-b border-white/5">
                    <div className="space-y-4">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-[10px] font-bold uppercase tracking-[0.2em] text-indigo-400">
                            <ShieldCheck className="w-3 h-3" />
                            Sesión Certificada / {jobId}
                        </div>
                        <h1 className="text-5xl md:text-6xl font-black tracking-tighter">
                            STUDIO <span className="text-white/20">PROGRESS</span>
                        </h1>
                        <p className="text-white/40 font-medium max-w-md">La IA está deconstruyendo tu audio para reconstruir una pieza visual profesional.</p>
                    </div>

                    <div className="text-right flex flex-col items-end gap-2">
                        <div className="text-7xl font-black tabular-nums tracking-tighter text-white">
                            {job.progress}<span className="text-xl text-indigo-500 font-bold ml-1">%</span>
                        </div>
                        <div className="flex items-center gap-2 text-white/20 text-[10px] font-bold uppercase tracking-widest">
                            <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                            Live Telemetry Stream
                        </div>
                    </div>
                </div>

                {/* 1. PROGRESS TRACKER */}
                <div className="space-y-8 bg-neutral-900/30 p-8 rounded-[40px] border border-white/5 backdrop-blur-3xl shadow-2xl">
                    <div className="relative h-2 w-full bg-white/5 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-1000 ease-out shadow-[0_0_20px_rgba(99,102,241,0.5)]"
                            style={{ width: `${job.progress}%` }}
                        />
                    </div>

                    <div className="grid grid-cols-4 md:grid-cols-7 gap-4">
                        {STAGES.map((stage, idx) => {
                            const isActive = idx <= currentStageIndex;
                            const isCurrent = idx === currentStageIndex;
                            const Icon = stage.icon;

                            return (
                                <div key={stage.status} className="flex flex-col items-center gap-4 group">
                                    <div className={cn(
                                        "w-12 h-12 rounded-2xl border flex items-center justify-center transition-all duration-700",
                                        isCurrent ? "bg-white border-white scale-110 shadow-[0_0_30px_rgba(255,255,255,0.3)]" :
                                            isActive ? "bg-indigo-500/20 border-indigo-500/40" :
                                                "bg-white/5 border-white/5 opacity-40 grayscale"
                                    )}>
                                        <Icon className={cn(
                                            "w-6 h-6",
                                            isCurrent ? "text-black" :
                                                isActive ? "text-indigo-400" : "text-white/40"
                                        )} />
                                    </div>
                                    <span className={cn(
                                        "text-[9px] font-black uppercase tracking-[0.2em] text-center hidden md:block transition-colors duration-500",
                                        isCurrent ? "text-white" : isActive ? "text-white/60" : "text-white/20"
                                    )}>
                                        {stage.label}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
                    {/* 2. ACTIVITY LOG & VIDEO PREVIEW */}
                    <div className="lg:col-span-2 space-y-8">
                        {/* Live Log */}
                        <div className="p-8 rounded-[40px] bg-neutral-900/30 border border-white/5 backdrop-blur-xl">
                            <div className="flex items-center justify-between mb-8">
                                <h3 className="text-sm font-black uppercase tracking-widest flex items-center gap-3">
                                    <Activity className="w-4 h-4 text-indigo-500" />
                                    Live Kernel Output
                                </h3>
                                <div className="px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-[8px] font-bold text-green-500 animate-pulse">
                                    ESTABLISHED
                                </div>
                            </div>

                            <div className="flex items-start gap-6 p-6 rounded-3xl bg-white/5 border border-white/5 border-l-4 border-l-indigo-500 animate-in fade-in slide-in-from-left-8 duration-500">
                                <div className="space-y-1 flex-1">
                                    <p className="text-lg font-bold text-white leading-tight">{job.stage}</p>
                                    <p className="text-[10px] text-white/30 uppercase tracking-[0.3em] font-black">Executing Pipeline Stage</p>
                                </div>
                                <ChevronRight className="w-5 h-5 text-indigo-500/50" />
                            </div>
                        </div>

                        {/* Video Result (Completed State) */}
                        {job.status === 'completed' && job.videoUrl ? (
                            <div className="p-1 rounded-[42px] bg-gradient-to-br from-indigo-500/50 via-purple-500/50 to-pink-500/50 group">
                                <div className="p-8 rounded-[40px] bg-black border border-white/10 overflow-hidden relative">
                                    <video
                                        src={job.videoUrl}
                                        controls
                                        autoPlay
                                        loop
                                        className="w-full aspect-[9/16] max-w-[360px] mx-auto rounded-3xl shadow-2xl ring-1 ring-white/20"
                                    />
                                    <div className="mt-8 flex flex-col items-center gap-6">
                                        <div className="text-center">
                                            <h4 className="text-2xl font-black tracking-tight">¡Producción Finalizada!</h4>
                                            <p className="text-white/40 text-sm font-medium">Renderizado en 1080p con hardware acceleration.</p>
                                        </div>
                                        <a
                                            href={job.videoUrl}
                                            download
                                            className="px-12 py-5 rounded-2xl bg-white text-black font-black uppercase text-xs tracking-[0.2em] hover:scale-105 transition-all shadow-xl"
                                        >
                                            Descargar Master MP4
                                        </a>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="aspect-video w-full rounded-[40px] bg-neutral-900/20 border border-white/5 border-dashed flex flex-col items-center justify-center text-white/10 gap-4">
                                <Clock className="w-12 h-12" />
                                <p className="font-black uppercase tracking-widest text-xs">Waiting for encoder output...</p>
                            </div>
                        )}
                    </div>

                    {/* 3. VOCAL DNA DASHBOARD */}
                    <div className="space-y-8">
                        <div className="p-8 rounded-[40px] bg-indigo-500/5 border border-indigo-500/10 backdrop-blur-3xl space-y-8">
                            <h3 className="text-sm font-black uppercase tracking-widest text-indigo-400 flex items-center gap-3">
                                <ShieldCheck className="w-4 h-4" />
                                Vocal DNA Metrics
                            </h3>

                            <div className="space-y-6">
                                {/* Metric: SNR */}
                                <div className="space-y-2">
                                    <div className="flex justify-between items-end">
                                        <span className="text-[10px] font-bold text-white/40 uppercase">Signal/Noise (SNR)</span>
                                        <span className="text-sm font-bold">{job.metrics?.snr?.toFixed(1) || '0.0'}<span className="text-[10px] text-white/20 ml-1">dB</span></span>
                                    </div>
                                    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                        <div className="h-full bg-indigo-500 transition-all duration-1000" style={{ width: `${Math.min(100, (job.metrics?.snr || 0) * 2)}%` }} />
                                    </div>
                                </div>

                                {/* Metric: Reverb */}
                                <div className="space-y-2">
                                    <div className="flex justify-between items-end">
                                        <span className="text-[10px] font-bold text-white/40 uppercase">Reverb Profile</span>
                                        <span className="text-sm font-bold">{((job.metrics?.reverbLevel || 0) * 100).toFixed(0)}<span className="text-[10px] text-white/20 ml-1">%</span></span>
                                    </div>
                                    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                        <div className="h-full bg-purple-500 transition-all duration-1000" style={{ width: `${(job.metrics?.reverbLevel || 0) * 100}%` }} />
                                    </div>
                                </div>

                                {/* Metric: Tempo */}
                                <div className="p-4 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-between">
                                    <span className="text-[10px] font-black text-white/30 uppercase">Detected Tempo</span>
                                    <span className="text-lg font-black text-white">{job.metrics?.tempo?.toFixed(0) || '--'}<span className="text-[10px] ml-1">BPM</span></span>
                                </div>

                                {/* Metadata: Lang */}
                                <div className="p-4 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-between">
                                    <span className="text-[10px] font-black text-white/30 uppercase">Target Engine</span>
                                    <span className="text-[10px] font-black uppercase text-indigo-400">{job.metadata?.targetLanguage || 'Native'}</span>
                                </div>
                            </div>

                            <p className="text-[10px] text-white/30 leading-relaxed italic">
                                * Nuestras métricas de SNR y Reverb determinan qué filtros de limpieza adaptativa (afftdn/adeclick) aplica el trabajador de audio.
                            </p>
                        </div>

                        {/* Style Card Mini */}
                        <div className="p-8 rounded-[40px] bg-white/5 border border-white/5 space-y-4">
                            <h4 className="text-[10px] font-black uppercase tracking-widest text-white/20">Selected Aesthetic</h4>
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-2xl bg-indigo-500/20 border border-indigo-500/20 flex items-center justify-center">
                                    <Activity className="w-6 h-6 text-indigo-400" />
                                </div>
                                <div>
                                    <p className="text-sm font-bold uppercase tracking-tight">{job.metadata?.styleId?.replace('_', ' ') || 'Tiktok Bold'}</p>
                                    <p className="text-[10px] font-bold text-white/20 uppercase tracking-widest">{job.metadata?.mood || 'Default Mood'}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
