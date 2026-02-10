'use client';

import React, { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface AudioWaveformProps {
    file: File;
    onRangeChange: (range: [number, number]) => void;
    maxDuration?: number; // default 60s
}

export function AudioWaveform({ file, onRangeChange, maxDuration = 60 }: AudioWaveformProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const [peaks, setPeaks] = useState<number[]>([]);
    const [duration, setDuration] = useState(0);
    const [range, setRange] = useState<[number, number]>([0, 0]);
    const [isResizing, setIsResizing] = useState<'left' | 'right' | 'move' | null>(null);

    // 1. Process Audio File
    useEffect(() => {
        const processAudio = async () => {
            const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

            setDuration(audioBuffer.duration);

            // Calculate initial range (max 60s)
            const initialEnd = Math.min(audioBuffer.duration, maxDuration);
            setRange([0, initialEnd]);
            onRangeChange([0, initialEnd]);

            // Calculate peaks for visualization
            const rawData = audioBuffer.getChannelData(0); // Use first channel
            const samples = 1200; // Increased Visualization resolution for full song fidelity
            const blockSize = Math.floor(rawData.length / samples);
            const filteredData = [];
            for (let i = 0; i < samples; i++) {
                let blockStart = blockSize * i;
                let sum = 0;
                for (let j = 0; j < blockSize; j++) {
                    sum = sum + Math.abs(rawData[blockStart + j]);
                }
                filteredData.push(sum / blockSize);
            }

            // Normalize peaks
            const multiplier = Math.pow(Math.max(...filteredData), -1);
            setPeaks(filteredData.map(n => n * multiplier));
        };

        processAudio();
    }, [file]);

    // 2. Draw Waveform
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || peaks.length === 0) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = canvas.offsetWidth * dpr;
        canvas.height = canvas.offsetHeight * dpr;
        ctx.scale(dpr, dpr);

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        const barWidth = width / peaks.length;

        ctx.clearRect(0, 0, width, height);

        // Draw background bars (dimmed but visible for full song context)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        peaks.forEach((peak, i) => {
            const h = peak * height * 0.7;
            const x = i * barWidth;
            // Draw a rounded-ish bar
            ctx.fillRect(x + 1, (height - h) / 2, Math.max(1, barWidth - 1), h);
        });

        // Draw active area bars (highlighted)
        const startIdx = Math.floor((range[0] / duration) * peaks.length);
        const endIdx = Math.floor((range[1] / duration) * peaks.length);

        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#6366f1'); // Indigo
        gradient.addColorStop(1, '#a855f7'); // Purple
        ctx.fillStyle = gradient;

        peaks.slice(startIdx, endIdx).forEach((peak, i) => {
            const h = peak * height * 0.7;
            const x = (startIdx + i) * barWidth;
            ctx.fillRect(x + 1, (height - h) / 2, Math.max(1, barWidth - 1), h);
        });
    }, [peaks, range, duration]);

    // 3. Handle Interactions
    const handleMouseDown = (type: 'left' | 'right' | 'move') => {
        setIsResizing(type);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isResizing || !containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        const time = (x / rect.width) * duration;

        let newRange: [number, number] = [...range];

        if (isResizing === 'left') {
            const potentialStart = Math.max(0, Math.min(time, range[1] - 0.5));
            // Ensure width doesn't exceed maxDuration
            if (range[1] - potentialStart <= maxDuration) {
                newRange[0] = potentialStart;
            } else {
                newRange[0] = range[1] - maxDuration;
            }
        } else if (isResizing === 'right') {
            const potentialEnd = Math.max(range[0] + 0.5, Math.min(time, duration));
            // Ensure width doesn't exceed maxDuration
            if (potentialEnd - range[0] <= maxDuration) {
                newRange[1] = potentialEnd;
            } else {
                newRange[1] = range[0] + maxDuration;
            }
        } else if (isResizing === 'move') {
            const currentWidth = range[1] - range[0];
            const start = Math.max(0, Math.min(time - currentWidth / 2, duration - currentWidth));
            newRange = [start, start + currentWidth];
        }

        // Limit selection to maxDuration only if we are RESIZING
        // If it's already over, we allow moving it? 
        // User said they want to crop the part they want.
        // I'll keep the duration constraint for the selection.

        setRange(newRange);
        onRangeChange(newRange);
    };

    useEffect(() => {
        const stop = () => setIsResizing(null);
        window.addEventListener('mouseup', stop);
        return () => window.removeEventListener('mouseup', stop);
    }, []);

    const formatTime = (t: number) => {
        const m = Math.floor(t / 60);
        const s = Math.floor(t % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    const selectionDuration = range[1] - range[0];
    const isDurationWarning = selectionDuration > maxDuration;

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-end px-2">
                <div className="flex gap-10">
                    <div className="flex flex-col gap-1">
                        <span className="text-[9px] font-black uppercase tracking-[0.2em] text-white/20">Segment Head</span>
                        <span className="text-2xl font-black text-white">{formatTime(range[0])}</span>
                    </div>
                    <div className="flex flex-col gap-1">
                        <span className="text-[9px] font-black uppercase tracking-[0.2em] text-white/20">Segment Tail</span>
                        <span className="text-2xl font-black text-white">{formatTime(range[1])}</span>
                    </div>
                </div>
                <div className="text-right flex flex-col items-end gap-1">
                    <span className="text-[9px] font-black uppercase tracking-[0.2em] text-white/20">Temporal Length</span>
                    <p className={cn(
                        "text-5xl font-black tracking-tighter transition-all duration-500",
                        isDurationWarning ? "text-red-500 scale-110" : "text-indigo-500"
                    )}>
                        {selectionDuration.toFixed(1)}<span className="text-xl ml-1 text-white/20">s</span>
                    </p>
                </div>
            </div>

            <div
                ref={containerRef}
                onMouseMove={handleMouseMove}
                className="relative h-48 w-full bg-white/[0.02] rounded-[40px] border border-white/5 overflow-hidden cursor-crosshair group shadow-inner"
            >
                {/* Full Audio Background Trace */}
                <canvas ref={canvasRef} className="w-full h-full pointer-events-none opacity-40" />

                {/* Selection Shadow - Left */}
                <div
                    className="absolute inset-y-0 left-0 bg-black/80 backdrop-blur-[4px] transition-all duration-300 pointer-events-none"
                    style={{ width: `${(range[0] / duration) * 100}%` }}
                >
                    <div className="absolute inset-y-0 right-0 w-px bg-white/10" />
                </div>

                {/* Selection Shadow - Right */}
                <div
                    className="absolute inset-y-0 right-0 bg-black/80 backdrop-blur-[4px] transition-all duration-300 pointer-events-none"
                    style={{ width: `${(1 - range[1] / duration) * 100}%` }}
                >
                    <div className="absolute inset-y-0 left-0 w-px bg-white/10" />
                </div>

                {/* Selection Window Interactor */}
                <div
                    onMouseDown={() => handleMouseDown('move')}
                    className={cn(
                        "absolute inset-y-0 cursor-grab active:cursor-grabbing transition-all duration-500",
                        isDurationWarning ? "bg-red-500/10" : "bg-indigo-500/5 hover:bg-indigo-500/10"
                    )}
                    style={{
                        left: `${(range[0] / duration) * 100}%`,
                        width: `${((range[1] - range[0]) / duration) * 100}%`
                    }}
                >
                    {/* Animated scanning line */}
                    <div className="absolute inset-y-0 left-0 w-px bg-gradient-to-b from-transparent via-white/20 to-transparent animate-pulse" />
                </div>

                {/* Precision Handles */}
                <div
                    onMouseDown={(e) => { e.stopPropagation(); handleMouseDown('left'); }}
                    className="absolute inset-y-0 w-1 cursor-ew-resize bg-white/40 hover:bg-white transition-colors group-handle-left z-30"
                    style={{ left: `${(range[0] / duration) * 100}%` }}
                >
                    <div className="absolute top-1/2 -translate-y-1/2 -right-3 w-6 h-12 bg-[#0A0A0A] rounded-2xl border border-white/20 flex flex-col items-center justify-center gap-1.5 shadow-xl">
                        <div className="w-0.5 h-3 bg-white/20 rounded-full" />
                        <div className="w-0.5 h-3 bg-white/20 rounded-full" />
                    </div>
                </div>

                <div
                    onMouseDown={(e) => { e.stopPropagation(); handleMouseDown('right'); }}
                    className="absolute inset-y-0 w-1 cursor-ew-resize bg-white/40 hover:bg-white transition-colors group-handle-right z-30"
                    style={{ left: `${(range[1] / duration) * 100}%` }}
                >
                    <div className="absolute top-1/2 -translate-y-1/2 -left-3 w-6 h-12 bg-[#0A0A0A] rounded-2xl border border-white/20 flex flex-col items-center justify-center gap-1.5 shadow-xl">
                        <div className="w-0.5 h-3 bg-white/20 rounded-full" />
                        <div className="w-0.5 h-3 bg-white/20 rounded-full" />
                    </div>
                </div>
            </div>

            <div className="flex items-center justify-between px-4">
                <p className="text-[10px] font-black uppercase tracking-[0.3em] text-white/20">
                    Master Duration: {formatTime(duration)}
                </p>
                {isDurationWarning && (
                    <div className="flex items-center gap-2 text-red-500 animate-pulse">
                        <div className="w-1.5 h-1.5 rounded-full bg-red-500" />
                        <span className="text-[10px] font-black uppercase tracking-widest">Exceso de duración (Max 60s sugeridos)</span>
                    </div>
                )}
            </div>
        </div>
    );
}
