'use client';

import React, { useState, useRef } from 'react';
import { Upload, FileAudio, CheckCircle2, AlertCircle, Loader2, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

import { StylePreviewer } from './style-previewer';

export function UploadForm() {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [prompt, setPrompt] = useState('');
    const [selectedStyleId, setSelectedStyleId] = useState('tiktok_bold');
    const [targetLanguage, setTargetLanguage] = useState(''); // Default empty (original)
    const fileInputRef = useRef<HTMLInputElement>(null);

    const LANGUAGES = [
        { id: '', name: 'Original (Sin traducción)' },
        { id: 'en', name: 'Inglés' },
        { id: 'es', name: 'Español' },
        { id: 'fr', name: 'Francés' },
        { id: 'de', name: 'Alemán' },
        { id: 'it', name: 'Italiano' },
        { id: 'pt', name: 'Portugués' },
        { id: 'ja', name: 'Japonés' },
        { id: 'zh-CN', name: 'Chino (Simplificado)' },
        { id: 'ko', name: 'Coreano' },
        { id: 'ru', name: 'Ruso' }
    ];

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile && selectedFile.type.startsWith('audio/')) {
            setFile(selectedFile);
            setError(null);
        } else {
            setError('Please select a valid audio file.');
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setProgress(10);
        setError(null);

        try {
            // 1. Get Signed URL from our API
            const signRes = await fetch('/api/v1/uploads/sign', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fileName: file.name,
                    contentType: file.type
                })
            });

            if (!signRes.ok) throw new Error('Failed to get upload URL');
            const { uploadUrl, assetPath } = await signRes.json();
            setProgress(30);

            // 2. Upload to Supabase Storage via Signed URL
            const uploadRes = await fetch(uploadUrl, {
                method: 'PUT',
                body: file,
                headers: {
                    'Content-Type': file.type
                }
            });

            if (!uploadRes.ok) throw new Error('Failed to upload file to storage');
            setProgress(70);

            // 3. Create Job in our API (Passing Prompt, StyleId and targetLanguage)
            const jobRes = await fetch('/api/v1/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: file.name,
                    audioUrl: assetPath, // Pass the path to the API for automatic asset creation
                    prompt: prompt,
                    styleId: selectedStyleId,
                    targetLanguage: targetLanguage || undefined, // Send if selected
                    startTime: 0,
                    endTime: 30
                })
            });

            if (!jobRes.ok) throw new Error('Failed to create production job');
            const { jobId } = await jobRes.json();

            setProgress(100);
            window.location.href = `/jobs/${jobId}`;

        } catch (err: any) {
            setError(err.message || 'An error occurred during upload.');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="w-full flex flex-col gap-12">

            {/* 1. VISUAL PREVIEWER SECTION */}
            <div className="w-full animate-in fade-in slide-in-from-top-8 duration-1000">
                <StylePreviewer
                    initialStyleId={selectedStyleId}
                    onStyleSelect={setSelectedId => setSelectedStyleId(setSelectedId)}
                />
            </div>

            {/* 2. UPLOAD & PROMPT SECTION */}
            <div className="w-full max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                <div className="p-8 rounded-3xl bg-black/40 backdrop-blur-2xl border border-white/10 shadow-2xl flex flex-col gap-6">
                    <div className="text-center space-y-2">
                        <h2 className="text-2xl font-bold text-white tracking-tight">Paso 1: Audio</h2>
                        <p className="text-white/60 text-sm">Sube tu archivo para procesar las letras</p>
                    </div>

                    <div
                        onClick={() => fileInputRef.current?.click()}
                        className={cn(
                            "w-full aspect-video rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-4 cursor-pointer transition-all duration-300",
                            file ? "border-indigo-500/50 bg-indigo-500/5" : "border-white/10 hover:border-white/20 bg-white/5 hover:bg-white/[0.08]"
                        )}
                    >
                        {file ? (
                            <div className="flex flex-col items-center gap-2 animate-in fade-in zoom-in duration-300">
                                <FileAudio className="w-12 h-12 text-indigo-400" />
                                <span className="text-indigo-200 font-medium line-clamp-1 px-4">{file.name}</span>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center gap-2 text-white/40">
                                <Upload className="w-8 h-8 mb-2" />
                                <p className="text-xs uppercase tracking-widest font-bold">Upload MP3/WAV</p>
                            </div>
                        )}
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="audio/*" className="hidden" />
                    </div>
                </div>

                <div className="p-8 rounded-3xl bg-black/40 backdrop-blur-2xl border border-white/10 shadow-2xl flex flex-col gap-6 h-full">
                    <div className="text-center space-y-2">
                        <h2 className="text-2xl font-bold text-white tracking-tight">Paso 2: Vibe & Idioma</h2>
                        <p className="text-white/60 text-sm">Describe el estilo y elige traducción</p>
                    </div>

                    <div className="flex flex-col gap-4">
                        <div className="relative group">
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder="Escribe aquí: Cyberpunk, futurista, romántico, movido..."
                                className="w-full h-24 bg-white/5 border border-white/10 rounded-2xl p-4 text-white placeholder:text-white/20 focus:outline-none focus:border-indigo-500/50 focus:ring-4 focus:ring-indigo-500/10 transition-all resize-none"
                            />
                            <div className="absolute bottom-4 right-4 text-[10px] font-bold text-indigo-400 uppercase tracking-widest flex items-center gap-2 opacity-50">
                                <Sparkles className="w-3 h-3" />
                                AI Optimized
                            </div>
                        </div>

                        {/* Language Selector */}
                        <div className="space-y-2">
                            <label className="text-[10px] font-bold text-white/40 uppercase tracking-widest px-1">Traducir letra a:</label>
                            <select
                                value={targetLanguage}
                                onChange={(e) => setTargetLanguage(e.target.value)}
                                className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white focus:outline-none focus:border-indigo-500/50 transition-all appearance-none cursor-pointer"
                            >
                                {LANGUAGES.map(lang => (
                                    <option key={lang.id} value={lang.id} className="bg-neutral-900 text-white">
                                        {lang.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <button
                            disabled={!file || uploading}
                            onClick={handleUpload}
                            className={cn(
                                "w-full py-4 rounded-xl font-bold text-sm tracking-widest uppercase transition-all duration-300 flex items-center justify-center gap-3",
                                file && !uploading
                                    ? "bg-white text-black hover:bg-neutral-200 shadow-[0_0_30px_rgba(255,255,255,0.2)]"
                                    : "bg-white/5 text-white/20 cursor-not-allowed border border-white/5"
                            )}
                        >
                            {uploading ? (
                                <><Loader2 className="w-4 h-4 animate-spin" /> {progress}%</>
                            ) : (
                                <><CheckCircle2 className="w-4 h-4" /> Crear Video Professional</>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {error && (
                <div className="w-full max-w-4xl mx-auto p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-start gap-3 text-red-400 animate-shake">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <p className="text-sm font-medium">{error}</p>
                </div>
            )}
        </div>
    );
}
