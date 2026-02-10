'use client';

import React from 'react';
import { cn } from '@/lib/utils';

interface RealisticMobileProps {
    children: React.ReactNode;
    className?: string;
    glowColor?: string;
}

export function RealisticMobile({ children, className, glowColor = '#6366f1' }: RealisticMobileProps) {
    return (
        <div className={cn("relative group perspective-1000", className)}>
            {/* External dynamic glow (reacts to content color) */}
            <div
                className="absolute -inset-10 blur-[100px] opacity-20 group-hover:opacity-30 transition-opacity duration-1000 pointer-events-none"
                style={{ background: `radial-gradient(circle, ${glowColor}55 0%, transparent 70%)` }}
            />

            {/* Main Body Frame */}
            <div className="relative w-[360px] h-[640px] bg-[#080808] rounded-[60px] p-3 shadow-[0_50px_100px_-20px_rgba(0,0,0,1),0_0_0_1px_rgba(255,255,255,0.05)] border border-white/10 ring-1 ring-white/5 overflow-hidden">

                {/* Bezel Interior (Slightly lighter than body) */}
                <div className="absolute inset-0 rounded-[58px] border-[8px] border-zinc-900/50 pointer-events-none z-20" />

                {/* Dynamic Island / Notch */}
                <div className="absolute top-6 left-1/2 -translate-x-1/2 w-24 h-7 bg-black rounded-full z-30 flex items-center justify-between px-4 ring-1 ring-white/5">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500/20" />
                    <div className="w-8 h-1 bg-white/5 rounded-full" />
                </div>

                {/* Side Buttons (Simulated) */}
                <div className="absolute top-24 -left-[2px] w-1 h-12 bg-zinc-800 rounded-r-lg border border-white/5 z-10" />
                <div className="absolute top-44 -left-[2px] w-1 h-16 bg-zinc-800 rounded-r-lg border border-white/5 z-10" />
                <div className="absolute top-36 -right-[2px] w-1 h-20 bg-zinc-800 rounded-l-lg border border-white/5 z-10" />

                {/* Screen Content Container */}
                <div className="relative w-full h-full bg-black rounded-[48px] overflow-hidden z-10">
                    {/* Surface Reflection (Glass effect) */}
                    <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/[0.03] to-white/[0.05] pointer-events-none z-50 rounded-[48px]" />

                    {/* The actual content (Video/Preview) */}
                    <div className="w-full h-full">
                        {children}
                    </div>
                </div>

                {/* Bottom Speaker/Charging Grills (Subtle) */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-4 opacity-50 z-20">
                    <div className="w-1 h-1 rounded-full bg-zinc-700" />
                    <div className="w-10 h-1 bg-zinc-800 rounded-full" />
                    <div className="w-1 h-1 rounded-full bg-zinc-700" />
                </div>
            </div>
        </div>
    );
}
