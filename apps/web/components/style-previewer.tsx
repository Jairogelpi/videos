'use client';

import React, { useState, useEffect } from 'react';
import { VISUAL_PRESETS, VisualPreset } from '@repo/shared';
import { cn } from '@/lib/utils';
import { Play, Sparkles } from 'lucide-react';

export function StylePreviewer({ onStyleSelect, initialStyleId }: { onStyleSelect: (id: string) => void, initialStyleId?: string }) {
    const [selectedId, setSelectedId] = useState(initialStyleId || 'tiktok_bold');
    const [previewText, setPreviewText] = useState('THE FUTURE OF LYRIC VIDEOS');
    const [isAnimating, setIsAnimating] = useState(true);

    const selectedStyle = VISUAL_PRESETS.find(s => s.id === selectedId) || VISUAL_PRESETS[0];

    useEffect(() => {
        onStyleSelect(selectedId);
    }, [selectedId, onStyleSelect]);

    // Simple loop to restart animation for preview
    useEffect(() => {
        const interval = setInterval(() => {
            setIsAnimating(false);
            setTimeout(() => setIsAnimating(true), 50);
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            {/* 1. MOCK SCREEN PREVIEW */}
            <div className="relative aspect-[9/16] w-full max-w-[320px] mx-auto rounded-[40px] border-[12px] border-neutral-900 bg-black overflow-hidden shadow-2xl ring-1 ring-white/10 group">
                {/* Mock Background Video (Placeholder Gradient) */}
                <div
                    className="absolute inset-0 transition-all duration-1000"
                    style={{
                        background: `radial-gradient(circle at center, ${selectedStyle.colors[0]}44 0%, #000 100%)`,
                    }}
                >
                    <div className="absolute inset-0 opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] brightness-100 contrast-150" />
                </div>

                {/* Lyrics Layer */}
                <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center">
                    {isAnimating && (
                        <div
                            className={cn(
                                "transition-all duration-500",
                                selectedStyle.animation === 'spring-scale' && "animate-in zoom-in-50 duration-300",
                                selectedStyle.animation === 'slide-up' && "animate-in slide-in-from-bottom-8 duration-500",
                                selectedStyle.animation === 'fade-in' && "animate-in fade-in duration-1000",
                                selectedStyle.animation === 'neon' && "animate-pulse",
                                selectedStyle.animation === 'glitch' && "animate-bounce" // Fallback CSS animation
                            )}
                            style={{
                                fontFamily: selectedStyle.fontFamily,
                                color: selectedStyle.colors[0],
                                fontSize: '2.5rem',
                                fontWeight: 900,
                                textShadow: `0 0 20px ${selectedStyle.colors[1]}66`,
                                textTransform: 'uppercase',
                                lineHeight: 1.1,
                            }}
                        >
                            {previewText}
                        </div>
                    )}
                </div>

                {/* TikTok UI Overlay Mock */}
                <div className="absolute bottom-8 left-4 right-12 flex flex-col gap-2 pointer-events-none opacity-40">
                    <div className="w-24 h-3 bg-white/20 rounded-full" />
                    <div className="w-48 h-3 bg-white/10 rounded-full" />
                </div>
                <div className="absolute right-4 bottom-24 flex flex-col gap-6 opacity-40">
                    <div className="w-10 h-10 rounded-full bg-white/20" />
                    <div className="w-10 h-10 rounded-full bg-white/20" />
                    <div className="w-10 h-10 rounded-full bg-white/20" />
                </div>
            </div>

            {/* 2. STYLE SELECTOR */}
            <div className="flex flex-col gap-6 h-full">
                <div className="space-y-2">
                    <h3 className="text-xl font-bold text-white flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-indigo-400" />
                        Librería de Estilos Profesionales
                    </h3>
                    <p className="text-white/40 text-sm">Selecciona una estética o deja que la IA elija por ti.</p>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 overflow-y-auto max-h-[500px] pr-2 custom-scrollbar">
                    {VISUAL_PRESETS.map((s) => (
                        <button
                            key={s.id}
                            onClick={() => setSelectedId(s.id)}
                            className={cn(
                                "p-4 rounded-2xl border transition-all duration-300 flex flex-col items-center gap-3 group relative overflow-hidden",
                                selectedId === s.id
                                    ? "bg-white/10 border-indigo-500/50 ring-2 ring-indigo-500/20"
                                    : "bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/[0.08]"
                            )}
                        >
                            <div
                                className="w-10 h-10 rounded-full blur-xl opacity-40 group-hover:opacity-60 transition-opacity"
                                style={{ background: `linear-gradient(45deg, ${s.colors[0]}, ${s.colors[1] || s.colors[0]})` }}
                            />
                            <div className="text-center">
                                <span className="block text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">{s.animation}</span>
                                <span className="text-sm font-bold text-white group-hover:text-indigo-300 transition-colors line-clamp-1">{s.name}</span>
                            </div>

                            {selectedId === s.id && (
                                <div className="absolute top-2 right-2">
                                    <div className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_10px_#6366f1]" />
                                </div>
                            )}
                        </button>
                    ))}
                </div>

                <div className="mt-auto p-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20">
                    <p className="text-xs text-indigo-200/60 leading-relaxed italic">
                        Tip: Escribe una vibra en el prompt (ej: "hyperpop energético") y la IA seleccionará la tipografía y animación perfecta.
                    </p>
                </div>
            </div>
        </div>
    );
}
