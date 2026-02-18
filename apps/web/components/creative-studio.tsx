'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
    Upload, FileAudio, CheckCircle2, AlertCircle,
    Loader2, Sparkles, Smartphone,
    ChevronRight, Languages, Music, Wand2,
    Layout, History, Clock, Check, Scissors, Settings2, ArrowLeft
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { VISUAL_PRESETS } from '@repo/shared';
import { AudioWaveform } from './audio-waveform';
import { RealisticMobile } from './realistic-mobile';

interface Job {
    id: string;
    title: string;
    status: string;
    progress: number;
    created_at: string;
}

export function CreativeStudio() {
    // 0. WORKFLOW STATE
    const [step, setStep] = useState(0); // 0: Ingest, 1: Sculpt, 2: Synth

    // 1. DATA STATE
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [jobs, setJobs] = useState<Job[]>([]);

    // Creative Controls
    const [prompt, setPrompt] = useState('');
    const [selectedStyleId, setSelectedStyleId] = useState(''); // '' = Auto (AI chooses based on prompt+music)
    const [customStyle, setCustomStyle] = useState('');
    const [targetLanguage, setTargetLanguage] = useState('');
    const [fontSize, setFontSize] = useState(6); // rem
    const [position, setPosition] = useState<'top' | 'center' | 'bottom'>('center');
    const [fontFamily, setFontFamily] = useState('Outfit');
    const [videoTitle, setVideoTitle] = useState('');
    const [titleFontFamily, setTitleFontFamily] = useState('Syne');
    const [animationEffect, setAnimationEffect] = useState('fade');
    const [lyricColor, setLyricColor] = useState('#ffffff');
    const [lyricOpacity, setLyricOpacity] = useState(1);
    const [timeRange, setTimeRange] = useState<[number, number]>([0, 30]);

    // Animation Loop State
    const [animationKey, setAnimationKey] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setAnimationKey(prev => prev + 1);
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const fileInputRef = useRef<HTMLInputElement>(null);

    const LANGUAGES = [
        { id: '', name: 'Original (Sin traducción)' },
        { id: 'en', name: 'Inglés' },
        { id: 'es', name: 'Español' },
        { id: 'fr', name: 'Francés' },
        { id: 'ja', name: 'Japonés' },
        { id: 'pt', name: 'Portugués' }
    ];

    const selectedStyle = selectedStyleId ? VISUAL_PRESETS.find(s => s.id === selectedStyleId) : null;

    useEffect(() => {
        if (selectedStyle) {
            setFontFamily(selectedStyle.fontFamily);
            setLyricColor(selectedStyle.colors[0]);
        }
    }, [selectedStyleId]);

    // 2. FETCH JOBS
    useEffect(() => {
        const fetchJobs = async () => {
            try {
                const res = await fetch('/api/v1/jobs');
                if (res.ok) {
                    const data = await res.json();
                    setJobs((data.jobs || []).slice(0, 15)); // Show last 15
                }
            } catch (e) {
                console.error('Failed to fetch jobs history', e);
            }
        };
        fetchJobs();
        const interval = setInterval(fetchJobs, 10000);
        return () => clearInterval(interval);
    }, []);

    // 3. HANDLERS
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile && selectedFile.type.startsWith('audio/')) {
            setFile(selectedFile);
            setError(null);
            setStep(1); // Auto-advance to trim step
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setError(null);

        try {
            const signRes = await fetch('/api/v1/uploads/sign', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fileName: file.name, contentType: file.type })
            });
            if (!signRes.ok) throw new Error('Failed to get upload URL');
            const { uploadUrl, assetPath } = await signRes.json();
            setProgress(30);

            const uploadRes = await fetch(uploadUrl, {
                method: 'PUT',
                body: file,
                headers: { 'Content-Type': file.type }
            });
            if (!uploadRes.ok) {
                const errText = await uploadRes.text();
                throw new Error(`Upload Failed: ${uploadRes.status} ${uploadRes.statusText} - ${errText.substring(0, 50)}`);
            }
            setProgress(60);

            const jobRes = await fetch('/api/v1/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: file.name,
                    audioUrl: assetPath,
                    prompt: prompt,
                    styleId: customStyle || selectedStyleId || undefined, // Prioritize custom text
                    targetLanguage: targetLanguage || undefined,
                    startTime: timeRange[0],
                    endTime: timeRange[1],
                    position: position,
                    fontSize: fontSize,
                    fontFamily: fontFamily,
                    videoTitle: videoTitle || file.name.split('.')[0],
                    titleFontFamily: titleFontFamily,
                    animationEffect: animationEffect,
                    lyricColor: lyricColor,
                    lyricOpacity: lyricOpacity
                })
            });

            if (!jobRes.ok) throw new Error('Failed to create production job');
            const { jobId } = await jobRes.json();
            setProgress(100);

            // Add directly to local state for immediate feedback
            setJobs(prev => [{ id: jobId, title: file.name, status: 'queued', progress: 0, created_at: new Date().toISOString() }, ...prev]);

            // Wait a bit then redirect or show success
            setTimeout(() => {
                window.location.href = `/jobs/${jobId}`;
            }, 1000);

        } catch (err: any) {
            setError(err.message || 'An error occurred.');
            setUploading(false);
        }
    };

    // 4. UI SECTIONS
    const renderSteps = () => (
        <div className="flex items-center justify-center gap-12 mb-20 select-none animate-in fade-in slide-in-from-top-4 duration-1000">
            {[
                { icon: Music, label: 'Ingest' },
                { icon: Scissors, label: 'Sculpt' },
                { icon: Settings2, label: 'Synth' }
            ].map((s, i) => (
                <React.Fragment key={i}>
                    <div className={cn(
                        "flex flex-col items-center gap-3 transition-all duration-700",
                        step === i ? "opacity-100" : "opacity-10 scale-90"
                    )}>
                        <div className={cn(
                            "w-12 h-12 rounded-2xl flex items-center justify-center border transition-all duration-500",
                            step === i ? "bg-white border-white text-black shadow-[0_0_30px_rgba(255,255,255,0.2)] rotate-3" : "bg-zinc-900 border-white/5 text-white/40"
                        )}>
                            <s.icon className="w-5 h-5" />
                        </div>
                        <span className="text-[10px] font-black uppercase tracking-[0.3em] font-display">{s.label}</span>
                    </div>
                    {i < 2 && <div className="w-16 h-[2px] bg-gradient-to-r from-white/10 to-transparent mt-[-24px]" />}
                </React.Fragment>
            ))}
        </div>
    );

    return (
        <div className="w-full max-w-[1600px] mx-auto py-12 px-6">

            {renderSteps()}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">

                {/* LEFT: History & Status (2 Cols) */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                        <History className="w-4 h-4 text-white/30" />
                        <span className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">Neural History</span>
                    </div>
                    <div className="space-y-3">
                        {jobs.length === 0 ? (
                            <div className="p-6 rounded-3xl border border-white/5 bg-white/2 opacity-20 text-center">
                                <span className="text-[9px] font-bold uppercase tracking-widest">Archive Empty</span>
                            </div>
                        ) : (
                            jobs.map(job => (
                                <div
                                    key={job.id}
                                    onClick={() => window.location.href = `/jobs/${job.id}`}
                                    className="p-4 rounded-3xl bg-zinc-900/50 border border-white/[0.03] space-y-2 group cursor-pointer hover:border-white/10 transition-all"
                                >
                                    <p className="text-[10px] font-black uppercase tracking-tight truncate opacity-40 group-hover:opacity-100 transition-opacity">{job.title}</p>
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-1.5">
                                            {job.status === 'completed' ? <Check className="w-2.5 h-2.5 text-green-500" /> : <Clock className="w-2.5 h-2.5 text-indigo-400 animate-pulse" />}
                                            <span className="text-[8px] font-bold uppercase tracking-widest text-white/10">{job.status}</span>
                                        </div>
                                        <span className="text-[8px] font-bold text-white/5">{job.progress}%</span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* CENTER: Main Workflow (Step Content) (6 Cols) */}
                <div className="lg:col-span-6 min-h-[600px]">

                    {/* STEP 0: INGEST */}
                    {step === 0 && (
                        <div className="h-full flex flex-col items-center justify-center space-y-8 animate-in fade-in zoom-in-95 duration-700">
                            <div className="text-center space-y-4">
                                <h2 className="text-4xl font-black uppercase tracking-tighter leading-none italic opacity-90 text-glow">
                                    Initialize Audio Master
                                </h2>
                                <p className="text-white/40 font-medium max-w-sm mx-auto uppercase text-[10px] tracking-[0.2em]">
                                    Sube tu pista de audio para comenzar la orquestación visual.
                                </p>
                            </div>

                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className="w-full max-w-md aspect-square rounded-[80px] border border-white/[0.03] bg-zinc-900/50 hover:bg-zinc-800/80 hover:border-white/10 transition-all duration-700 cursor-pointer flex flex-col items-center justify-center group shadow-2xl relative overflow-hidden"
                            >
                                <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-transparent via-white/5 to-transparent scale-x-0 group-hover:scale-x-100 transition-transform duration-700" />
                                <div className="w-28 h-28 rounded-full bg-black flex items-center justify-center mb-8 ring-1 ring-white/5 group-hover:ring-white/10 group-hover:scale-105 transition-all duration-700">
                                    <Upload className="w-8 h-8 text-white/10 group-hover:text-white group-hover:animate-bounce" />
                                </div>
                                <span className="text-[11px] font-black uppercase tracking-[0.5em] text-white/10 group-hover:text-white/60 transition-colors">Select Master file</span>
                                <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="audio/*" className="hidden" />
                            </div>
                        </div>
                    )}

                    {/* STEP 1: SCULPT */}
                    {step === 1 && file && (
                        <div className="space-y-10 animate-in fade-in slide-in-from-right-8 duration-700">
                            <div className="flex items-center justify-between border-b border-white/5 pb-8">
                                <div className="space-y-2">
                                    <h2 className="text-3xl font-black uppercase tracking-tighter text-glow">Temporal Sculpting</h2>
                                    <p className="text-[10px] font-black uppercase tracking-widest text-indigo-400/60">Audio: {file.name}</p>
                                </div>
                                <button onClick={() => setStep(0)} className="p-3 rounded-full bg-white/5 hover:bg-white/10 text-white/40 transition-all">
                                    <ArrowLeft className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="bg-zinc-900/80 border border-white/[0.03] p-12 rounded-[56px] shadow-2xl ring-1 ring-white/5">
                                <AudioWaveform
                                    file={file}
                                    onRangeChange={setTimeRange}
                                />
                            </div>

                            <div className="pt-8 flex justify-end">
                                <button
                                    disabled={timeRange[1] - timeRange[0] > 60.5}
                                    onClick={() => setStep(2)}
                                    className={cn(
                                        "px-12 py-5 rounded-full font-black uppercase tracking-[0.3em] text-[10px] transition-all shadow-xl flex items-center gap-4",
                                        timeRange[1] - timeRange[0] > 60.5
                                            ? "bg-zinc-800 text-white/20 cursor-not-allowed"
                                            : "bg-white text-black hover:scale-105 shadow-white/5"
                                    )}
                                >
                                    Confirm & Continue
                                    <ChevronRight className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    )}

                    {/* STEP 2: SYNTH */}
                    {step === 2 && (
                        <div className="space-y-10 animate-in fade-in slide-in-from-right-8 duration-700">
                            <div className="flex items-center justify-between border-b border-white/5 pb-8">
                                <div className="space-y-2">
                                    <h2 className="text-3xl font-black uppercase tracking-tighter text-glow">Cinematic Synthesis</h2>
                                    <p className="text-[10px] font-black uppercase tracking-widest text-purple-400/60">Configurando {file?.name}</p>
                                </div>
                                <button onClick={() => setStep(1)} className="p-3 rounded-full bg-white/5 hover:bg-white/10 text-white/40 transition-all">
                                    <ArrowLeft className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="grid grid-cols-1 gap-8">
                                {/* Prompt Engine */}
                                <div className="p-8 rounded-[56px] bg-zinc-900/50 border border-white/[0.03] space-y-6">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-purple-500/10 flex items-center justify-center">
                                            <Sparkles className="w-4 h-4 text-purple-400" />
                                        </div>
                                        <h3 className="text-sm font-black uppercase tracking-widest text-white/90">Master Instruction</h3>
                                    </div>
                                    <div className="relative group">
                                        <textarea
                                            value={prompt}
                                            onChange={(e) => setPrompt(e.target.value)}
                                            placeholder="Define visual DNA: 'A world of liquid glass and chrome'..."
                                            className="w-full h-40 bg-black border border-white/[0.03] rounded-3xl p-8 text-base text-white placeholder:text-white/5 focus:outline-none focus:border-purple-500/20 transition-all resize-none shadow-inner"
                                        />
                                    </div>
                                    <div className="flex items-center justify-between p-6 rounded-3xl bg-black/40 border border-white/[0.03]">
                                        <span className="text-[10px] font-black uppercase tracking-widest text-white/20">Target Dialect</span>
                                        <select
                                            value={targetLanguage}
                                            onChange={(e) => setTargetLanguage(e.target.value)}
                                            className="bg-black border border-white/5 rounded-xl px-4 py-2 text-[10px] font-black uppercase text-indigo-400 focus:outline-none"
                                        >
                                            {LANGUAGES.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
                                        </select>
                                    </div>

                                    {/* Video Title Input */}
                                    <div className="space-y-4 pt-4 border-t border-white/5">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded-full bg-indigo-500/10 flex items-center justify-center">
                                                <Smartphone className="w-4 h-4 text-indigo-400" />
                                            </div>
                                            <h3 className="text-sm font-black uppercase tracking-widest text-white/90">Main Title Overlay</h3>
                                        </div>
                                        <input
                                            type="text"
                                            value={videoTitle}
                                            onChange={(e) => setVideoTitle(e.target.value)}
                                            placeholder="Enter Video Title (e.g. Song Name)"
                                            className="w-full bg-black border border-white/[0.03] rounded-2xl px-6 py-4 text-sm text-white focus:outline-none focus:border-indigo-500/30 transition-all"
                                        />

                                        <div className="space-y-2">
                                            <span className="text-[10px] font-black uppercase tracking-widest text-white/20 px-1">Title Typography</span>
                                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                                {['Syne', 'Anton', 'Unbounded', 'Bebas Neue', 'Righteous', 'Michroma'].map((f) => (
                                                    <button
                                                        key={f}
                                                        onClick={() => setTitleFontFamily(f)}
                                                        className={cn(
                                                            "py-3 px-4 rounded-xl border text-[9px] font-black uppercase tracking-[0.1em] transition-all duration-300 text-left truncate",
                                                            titleFontFamily === f ? "bg-indigo-500 border-indigo-400 text-white" : "bg-black/50 border-white/[0.03] text-white/20 hover:border-white/10"
                                                        )}
                                                    >
                                                        {f}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Visual Style (Optional) */}
                                <div className="p-8 rounded-[56px] bg-zinc-900/50 border border-white/[0.03] space-y-6">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-cyan-500/10 flex items-center justify-center">
                                            <Sparkles className="w-4 h-4 text-cyan-400" />
                                        </div>
                                        <h3 className="text-sm font-black uppercase tracking-widest text-white/90">Visual Style</h3>
                                        <span className="text-[9px] font-bold uppercase tracking-wider text-white/20 ml-auto">Opcional</span>
                                    </div>
                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                        {/* Auto option */}
                                        <button
                                            onClick={() => setSelectedStyleId('')}
                                            className={cn(
                                                "py-4 px-5 rounded-2xl border text-[10px] font-black uppercase tracking-[0.1em] transition-all duration-300 text-left",
                                                selectedStyleId === ''
                                                    ? "bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border-purple-500/30 text-white shadow-lg"
                                                    : "bg-black/50 border-white/[0.03] text-white/20 hover:border-white/10"
                                            )}
                                        >
                                            <span className="flex items-center gap-2">
                                                <Wand2 className="w-3 h-3" />
                                                Auto (IA)
                                            </span>
                                        </button>
                                        {VISUAL_PRESETS.map((s) => (
                                            <button
                                                key={s.id}
                                                onClick={() => setSelectedStyleId(s.id)}
                                                className={cn(
                                                    "py-4 px-5 rounded-2xl border text-[10px] font-black uppercase tracking-[0.1em] transition-all duration-300 text-left truncate",
                                                    selectedStyleId === s.id
                                                        ? "bg-white border-white text-black"
                                                        : "bg-black/50 border-white/[0.03] text-white/20 hover:border-white/10"
                                                )}
                                            >
                                                <span className="flex items-center gap-2">
                                                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: `linear-gradient(135deg, ${s.colors[0]}, ${s.colors[1]})` }} />
                                                    {s.name}
                                                </span>
                                            </button>
                                        ))}
                                    </div>

                                    {/* Custom Style Text Input */}
                                    <div className="space-y-4 pt-4 border-t border-white/5">
                                        <div className="flex justify-between items-center px-1">
                                            <span className="text-[10px] font-black uppercase tracking-widest text-white/20">Custom Artistic Movement</span>
                                            <span className="text-[8px] font-bold uppercase tracking-widest text-white/5">Overrides presets</span>
                                        </div>
                                        <input
                                            type="text"
                                            value={customStyle}
                                            onChange={(e) => setCustomStyle(e.target.value)}
                                            placeholder="e.g. Cyber-Punk Baroque, Wes Anderson style, etc."
                                            className="w-full bg-black border border-white/[0.03] rounded-2xl px-6 py-4 text-sm text-white focus:outline-none focus:border-cyan-500/30 transition-all"
                                        />
                                    </div>
                                </div>

                                {/* Lyrical Geometry (Size & Position) */}
                                <div className="p-8 rounded-[56px] bg-zinc-900/50 border border-white/[0.03] space-y-8">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-orange-500/10 flex items-center justify-center">
                                            <Layout className="w-4 h-4 text-orange-400" />
                                        </div>
                                        <h3 className="text-sm font-black uppercase tracking-widest text-white/90">Lyrical Geometry</h3>
                                    </div>

                                    <div className="space-y-6">
                                        {/* Font Size Slider */}
                                        <div className="space-y-4">
                                            <div className="flex justify-between items-center px-1">
                                                <span className="text-[10px] font-black uppercase tracking-widest text-white/20">Lyrical Scale</span>
                                                <span className="text-xl font-black text-white">{fontSize.toFixed(1)} <span className="text-[10px] text-white/20 ml-1">REM</span></span>
                                            </div>
                                            <input
                                                type="range"
                                                min="2"
                                                max="12"
                                                step="0.5"
                                                value={fontSize}
                                                onChange={(e) => setFontSize(parseFloat(e.target.value))}
                                                className="w-full h-1 bg-white/5 rounded-full appearance-none cursor-pointer accent-orange-500"
                                            />
                                        </div>

                                        {/* Position Selector */}
                                        <div className="space-y-4">
                                            <span className="text-[10px] font-black uppercase tracking-widest text-white/20 px-1">Screen Placement</span>
                                            <div className="grid grid-cols-3 gap-2">
                                                {(['top', 'center', 'bottom'] as const).map((p) => (
                                                    <button
                                                        key={p}
                                                        onClick={() => setPosition(p)}
                                                        className={cn(
                                                            "py-3 rounded-2xl border text-[10px] font-black uppercase tracking-[0.2em] transition-all duration-300",
                                                            position === p ? "bg-white border-white text-black shadow-lg" : "bg-black/50 border-white/[0.03] text-white/20 hover:border-white/10"
                                                        )}
                                                    >
                                                        {p}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Opacity Slider */}
                                        <div className="space-y-4">
                                            <div className="flex justify-between items-center px-1">
                                                <span className="text-[10px] font-black uppercase tracking-widest text-white/20">Lyrical Opacity</span>
                                                <span className="text-xl font-black text-white">{Math.round(lyricOpacity * 100)}%</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="0.1"
                                                max="1"
                                                step="0.05"
                                                value={lyricOpacity}
                                                onChange={(e) => setLyricOpacity(parseFloat(e.target.value))}
                                                className="w-full h-1 bg-white/5 rounded-full appearance-none cursor-pointer accent-white"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Appearance Engine */}
                                <div className="p-8 rounded-[56px] bg-zinc-900/50 border border-white/[0.03] space-y-8">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center">
                                            <Wand2 className="w-4 h-4 text-blue-400" />
                                        </div>
                                        <h3 className="text-sm font-black uppercase tracking-widest text-white/90">Appearance Engine</h3>
                                    </div>

                                    <div className="space-y-8">
                                        {/* Color Selector */}
                                        <div className="space-y-4">
                                            <span className="text-[10px] font-black uppercase tracking-widest text-white/20 px-1">Lyrical Pigment</span>
                                            <div className="flex flex-wrap gap-2">
                                                {[
                                                    '#ffffff', // White
                                                    '#6366f1', // Indigo
                                                    '#a855f7', // Purple
                                                    '#ec4899', // Pink
                                                    '#f87171', // Red
                                                    '#fbbf24', // Amber
                                                    '#34d399', // Emerald
                                                    '#22d3ee', // Cyan
                                                    '#fb923c', // Orange
                                                    '#818cf8', // Periwinkle
                                                    '#c084fc', // Orchid
                                                    '#f472b6', // Rose
                                                    '#4ade80', // Mint
                                                    '#facc15', // Yellow
                                                ].map((c) => (
                                                    <button
                                                        key={c}
                                                        onClick={() => setLyricColor(c)}
                                                        className={cn(
                                                            "w-10 h-10 rounded-full border-2 transition-all duration-300",
                                                            lyricColor === c ? "border-white scale-110 shadow-lg" : "border-transparent opacity-40 hover:opacity-100"
                                                        )}
                                                        style={{ backgroundColor: c }}
                                                    />
                                                ))}
                                                <div className="relative">
                                                    <input
                                                        type="color"
                                                        value={lyricColor}
                                                        onChange={(e) => setLyricColor(e.target.value)}
                                                        className="w-10 h-10 rounded-full bg-transparent border-2 border-white/[0.05] p-0 cursor-pointer overflow-hidden"
                                                    />
                                                </div>
                                            </div>
                                        </div>

                                        {/* Font Selector */}
                                        <div className="space-y-4">
                                            <span className="text-[10px] font-black uppercase tracking-widest text-white/20 px-1">Typography DNA</span>
                                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                                {[
                                                    'Outfit', 'Inter', 'Montserrat', 'Syne',
                                                    'Anton', 'Bebas Neue', 'Unbounded',
                                                    'Space Grotesk', 'Playfair Display',
                                                    'Righteous', 'Permanent Marker', 'Lobster',
                                                    'Cinzel', 'Michroma', 'Quicksand'
                                                ].map((f) => (
                                                    <button
                                                        key={f}
                                                        onClick={() => setFontFamily(f)}
                                                        className={cn(
                                                            "py-4 px-6 rounded-2xl border text-[10px] font-black uppercase tracking-[0.1em] transition-all duration-300 text-left truncate",
                                                            fontFamily === f ? "bg-white border-white text-black" : "bg-black/50 border-white/[0.03] text-white/20 hover:border-white/10"
                                                        )}
                                                        style={{ fontFamily: f.toLowerCase().replace(/\s+/g, '-') }}
                                                    >
                                                        {f}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Animation Selector */}
                                        <div className="space-y-4">
                                            <span className="text-[10px] font-black uppercase tracking-widest text-white/20 px-1">Entrance Kinetic</span>
                                            <div className="grid grid-cols-2 gap-2">
                                                {['fade', 'pop', 'slide', 'typewriter', 'glitch', 'neon', 'shake', 'kinetic'].map((a) => (
                                                    <button
                                                        key={a}
                                                        onClick={() => setAnimationEffect(a)}
                                                        className={cn(
                                                            "py-4 px-6 rounded-2xl border text-[10px] font-black uppercase tracking-[0.1em] transition-all duration-300 text-left",
                                                            animationEffect === a ? "bg-white border-white text-black" : "bg-black/50 border-white/[0.03] text-white/20 hover:border-white/10"
                                                        )}
                                                    >
                                                        {a}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <button
                                disabled={uploading}
                                onClick={handleUpload}
                                className={cn(
                                    "w-full py-10 rounded-[56px] font-black uppercase text-base tracking-[0.6em] transition-all duration-700 relative overflow-hidden ring-1 ring-white/10",
                                    !uploading ? "bg-white text-black hover:tracking-[0.8em] shadow-[0_0_80px_rgba(255,255,255,0.1)]" : "bg-zinc-900 text-white/10"
                                )}
                            >
                                {uploading ? (
                                    <div className="flex items-center justify-center gap-6">
                                        <Loader2 className="w-6 h-6 animate-spin text-white/20" />
                                        <span>SYNTHESIZING... {progress}%</span>
                                    </div>
                                ) : (
                                    <span>INITIATE PRODUCTION</span>
                                )}
                            </button>
                        </div>
                    )}
                </div>

                {/* RIGHT: Real-time Mockup (4 Cols) */}
                <div className="lg:col-span-4 sticky top-12 space-y-8 flex flex-col items-center">
                    <div className="flex items-center justify-between w-full opacity-20 px-8">
                        <span className="text-[10px] font-black uppercase tracking-[0.3em]">Live Master Output</span>
                        <div className="flex gap-1">
                            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                            <div className="w-1.5 h-1.5 rounded-full bg-green-500/40" />
                        </div>
                    </div>

                    <RealisticMobile glowColor={lyricColor}>
                        <div className="relative w-full h-full bg-black overflow-hidden group/screen">
                            {/* Animated Background simulation */}
                            <div
                                className="absolute inset-0 transition-all duration-1000 scale-110 group-hover/screen:scale-100"
                                style={{
                                    background: `radial-gradient(circle at center, ${lyricColor}15 0%, #000 100%)`
                                }}
                            />

                            {/* Content Layer with Camera Motion simulation */}
                            <div className={cn(
                                "absolute inset-0 z-20 flex flex-col items-center text-center transition-all duration-1000 animate-float",
                                position === 'top' ? "justify-start" : position === 'bottom' ? "justify-end" : "justify-center"
                            )}
                                style={{
                                    paddingTop: position === 'top' ? '64px' : '0', // 192px / 3 (pt-48)
                                    paddingBottom: position === 'bottom' ? '85px' : '0', // 256px / 3 (pb-64)
                                    paddingLeft: '21px', // 64px / 3 (px-16)
                                    paddingRight: '21px'
                                }}>
                                <div
                                    className={cn(
                                        "text-glow transition-all duration-700 relative",
                                        animationEffect === 'neon' ? 'animate-neon' :
                                            animationEffect === 'shake' ? 'animate-shake-viral' :
                                                animationEffect === 'kinetic' ? 'animate-kinetic' :
                                                    animationEffect === 'glitch' ? 'animate-glitch' :
                                                        animationEffect === 'fade' ? 'animate-fade' :
                                                            animationEffect === 'pop' ? 'animate-pop' :
                                                                animationEffect === 'slide' ? 'animate-slide' :
                                                                    animationEffect === 'typewriter' ? 'animate-typewriter' : ''
                                    )}
                                    // Key forces remount to replay animation entrance
                                    key={`${animationEffect}-${animationKey}`}
                                    style={{
                                        fontFamily: `var(--font-${fontFamily.toLowerCase()})`,
                                        color: lyricColor,
                                        fontSize: `${fontSize * 0.333}rem`,
                                        fontWeight: 900,
                                        opacity: lyricOpacity,
                                        textTransform: 'uppercase',
                                        lineHeight: 0.85,
                                        filter: `drop-shadow(0 0 30px ${lyricColor}44)`
                                    }}
                                >
                                    {/* Ghost Trail implementation for mockup */}
                                    {animationEffect === 'glitch' && (
                                        <div className="ghost-layer" style={{ fontFamily: `var(--font-${fontFamily.toLowerCase()})`, color: lyricColor }}>
                                            NEURAL BRIDGE
                                        </div>
                                    )}
                                    NEURAL<br />
                                    <span className="text-white opacity-40 italic font-light lowercase tracking-tighter" style={{ fontFamily }}>BRIDGE</span><br />
                                    ORCHESTRATION
                                </div>
                            </div>

                            {/* UI Overlay simulation */}
                            <div className="absolute bottom-16 inset-x-0 px-10 z-20 space-y-6 opacity-30">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-full bg-white/10" />
                                    <div className="space-y-2">
                                        <div className="w-24 h-2 bg-white/20 rounded-full" />
                                        <div className="w-16 h-1.5 bg-white/10 rounded-full" />
                                    </div>
                                </div>
                                <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                                    <div className="h-full bg-white/40" style={{ width: '65%' }} />
                                </div>
                            </div>
                        </div>
                    </RealisticMobile>

                    {error && (
                        <div className="p-8 rounded-[32px] bg-red-900/10 border border-red-500/20 text-red-500 flex items-center gap-5 glass-card animate-shake">
                            <AlertCircle className="w-5 h-5 flex-shrink-0" />
                            <p className="text-[10px] font-black uppercase tracking-widest">{error}</p>
                        </div>
                    )}
                </div>
            </div>
        </div >
    );
}
