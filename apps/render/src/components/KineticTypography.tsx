import React, { useMemo } from 'react';
import { interpolate, spring, Easing } from 'remotion';
import { MotionManifest } from '../types/MotionManifest';

interface KineticTypographyProps {
    text: string;
    manifest: MotionManifest;
    energy: number; // Normalized 0-1
    t: number; // Time since appearance (frames)
    fps: number;
}

export const KineticTypography: React.FC<KineticTypographyProps> = ({
    text,
    manifest,
    energy,
    t,
    fps
}) => {
    const { typography, palette, kinetics } = manifest;

    // --- PHYSICS ENGINE --- //
    // 1. Scales based on Energy (Reactivity)
    const reactiveScale = 1 + (energy * (kinetics.reactivity || 0) * 0.1);

    // 2. Jitter (Shake)
    const jitterX = (Math.random() - 0.5) * (kinetics.jitter || 0) * 20 * energy;
    const jitterY = (Math.random() - 0.5) * (kinetics.jitter || 0) * 20 * energy;

    // 3. Entrance Animation (Spring / Linear)
    const entranceProgress = spring({
        frame: t,
        fps,
        config: {
            damping: kinetics.physics === 'bounce' ? 8 : 12,
            stiffness: kinetics.physics === 'elastic' ? 200 : 100
        }
    });

    // 4. Effect Logic
    const effectStyle = useMemo((): React.CSSProperties => {
        const style: React.CSSProperties = {};

        switch (kinetics.effect) {
            case 'typewriter':
                // Logic handled in render content
                break;
            case 'fade':
                style.opacity = interpolate(t, [0, 10], [0, 1], { extrapolateRight: 'clamp' });
                break;
            case 'slide':
                const slideY = interpolate(t, [0, 15], [80, 0], { extrapolateRight: 'clamp', easing: Easing.out(Easing.back(1.5)) });
                style.transform = `translateY(${slideY}px)`;
                break;
            case 'pop':
                style.transform = `scale(${entranceProgress})`;
                break;
            case 'glitch':
                if (energy > 0.6) {
                    const offset = (Math.random() - 0.5) * 40 * energy;
                    style.transform = `translate(${offset}px, 0) skew(${offset}deg)`;
                    style.filter = `hue-rotate(${t * 10}deg) brightness(${1 + energy})`;
                    style.textShadow = `${offset}px 0 ${palette.secondary}, ${-offset}px 0 ${palette.glow || 'cyan'}`;
                }
                break;
            case 'neon':
                const pulse = Math.sin(t / 4) * 0.5 + 0.5;
                style.textShadow = `0 0 ${15 + pulse * 30}px ${palette.glow || palette.primary}, 0 0 ${10}px ${palette.secondary}`;
                style.opacity = 0.8 + (pulse * 0.2);
                break;
            case 'shake':
                const freq = 0.5 + energy;
                const ampl = 20 * energy;
                style.transform = `translate(${Math.sin(t * freq) * ampl}px, ${Math.cos(t * freq * 1.1) * ampl}px) rotate(${Math.sin(t * 0.1) * 5}deg)`;
                break;
            case 'kinetic':
                const rotation = interpolate(t, [0, 30], [20, 0], { extrapolateRight: 'clamp' });
                const scale = interpolate(t, [0, 10], [0, 1], { easing: Easing.out(Easing.exp) });
                style.transform = `rotate(${rotation}deg) scale(${scale * reactiveScale})`;
                style.letterSpacing = `${interpolate(t, [0, 20], [1, 0], { extrapolateRight: 'clamp' })}em`;
                break;
        }
        return style;
    }, [kinetics.effect, t, energy, palette, entranceProgress, reactiveScale]);

    // Apply User Constraints (Explicitly Overriding Defaults)
    const baseStyle: React.CSSProperties = {
        fontFamily: typography.fontFamily,
        fontWeight: typography.fontWeight,
        fontSize: `${typography.fontSize || 6}rem`,
        color: palette.primary,
        textShadow: effectStyle.textShadow || `0 4px 20px ${palette.shadow || 'rgba(0,0,0,0.5)'}`,
        letterSpacing: effectStyle.letterSpacing || `${typography.tracking || 0}em`,
        opacity: (typography.opacity ?? 1) * ((effectStyle.opacity as number) ?? 1),
        transform: `
            ${effectStyle.transform || ''}
            translate(${jitterX}px, ${jitterY}px)
        `,
        filter: effectStyle.filter,
        textAlign: 'center',
        padding: '0 2rem'
    };

    // Special handling for Typewriter effect text content
    const displayText = kinetics.effect === 'typewriter'
        ? text.substring(0, Math.floor(interpolate(t, [0, Math.max(10, text.length * 1.5)], [0, text.length], { extrapolateRight: 'clamp' })))
        : text;

    return (
        <div style={baseStyle} className="kinetic-typography">
            {displayText}
        </div>
    );
};
