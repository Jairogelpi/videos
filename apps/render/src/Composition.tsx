import React, { useMemo } from 'react';
import { useCurrentFrame, useVideoConfig, Audio, AbsoluteFill, OffthreadVideo, interpolate } from 'remotion';
import { KineticTypography } from './components/KineticTypography';
import { MotionManifest } from './types/MotionManifest';

interface Word {
  t0: number;
  t1: number;
  w: string;
  conf?: number;
  lang?: string;
  tw?: string;
  type?: 'lyric' | 'phrase' | 'caption';
}

interface UniversalProps {
  words: Word[];
  audioUrl: string;
  videoBgUrl?: string;
  hook?: { start: number; end: number };
  energy_flux?: number[];
  motion_manifest?: MotionManifest;
  // Legacy props as fallbacks
  styleId?: string;
  mood?: string;
  position?: string;
  fontSize?: number;
  fontFamily?: string;
  lyricColor?: string;
  lyricOpacity?: number;
  animationEffect?: string;
  videoTitle?: string;
  titleFontFamily?: string;
}

export const MyComposition: React.FC<UniversalProps> = ({
  words = [],
  audioUrl,
  videoBgUrl,
  hook,
  energy_flux = [],
  motion_manifest,
  // Fallbacks for legacy jobs
  styleId, position, fontSize, fontFamily, lyricColor, lyricOpacity, animationEffect,
  videoTitle, titleFontFamily
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentTime = frame / fps;

  // 1. Resolve Motion Manifest (Safe Deep Merge)
  const manifest: MotionManifest = useMemo(() => {
    const defaultManifest: MotionManifest = {
      typography: {
        fontFamily: fontFamily || 'Inter',
        fontWeight: 800,
        fontSize: fontSize || 6,
        opacity: lyricOpacity ?? 1
      },
      palette: {
        primary: lyricColor || '#ffffff',
        secondary: '#cccccc',
        shadow: 'rgba(0,0,0,0.5)',
        glow: 'transparent'
      },
      layout: { mode: (position as any) || 'center' },
      kinetics: {
        reactivity: 1.0,
        jitter: 0.0,
        effect: animationEffect || 'fade',
        physics: 'spring'
      }
    };

    if (!motion_manifest) return defaultManifest;

    // Merge logic: prefer manifest but fallback to defaults or props
    return {
      typography: {
        ...defaultManifest.typography,
        ...motion_manifest.typography,
        // Override with explicit props if they exist
        ...(fontFamily ? { fontFamily } : {}),
        ...(fontSize ? { fontSize } : {}),
        ...(lyricOpacity !== undefined ? { opacity: lyricOpacity } : {}),
      },
      palette: {
        ...defaultManifest.palette,
        ...motion_manifest.palette,
        ...(lyricColor ? { primary: lyricColor } : {}),
      },
      layout: {
        ...defaultManifest.layout,
        ...motion_manifest.layout,
        ...(position ? { mode: position as any } : {}),
      },
      kinetics: {
        ...defaultManifest.kinetics,
        ...motion_manifest.kinetics,
        ...(animationEffect ? { effect: animationEffect } : {}),
      }
    };
  }, [motion_manifest, fontFamily, fontSize, lyricOpacity, position, animationEffect, lyricColor]);

  // 1a. Dynamic Font Loading (Filter out AI prompts or long descriptions)
  const isValidFont = (f: string) => f && f.length < 40 && !f.includes('(') && !f.includes('Choose');
  const fontsToLoad = Array.from(new Set([
    isValidFont(manifest.typography.fontFamily) ? manifest.typography.fontFamily : 'Inter',
    isValidFont(titleFontFamily || '') ? titleFontFamily : 'Syne'
  ])).filter(Boolean) as string[];

  const fontUrl = `https://fonts.googleapis.com/css2?family=${fontsToLoad.map(f => f.replace(/\s+/g, '+')).join('&family=')}:wght@400;700;900&display=swap`;

  // 2. Audio Energy Calculation (Normalize CamelCase vs snake_case)
  const activeEnergySeries = (energy_flux && energy_flux.length > 0)
    ? energy_flux
    : (manifest as any).energyFlux || [];

  const currentSecond = Math.floor(currentTime);
  const localEnergy = activeEnergySeries[currentSecond] || 0.5;
  const nextEnergy = activeEnergySeries[currentSecond + 1] || localEnergy;
  const energyProgress = currentTime % 1;
  const smoothEnergy = localEnergy * (1 - energyProgress) + nextEnergy * energyProgress;

  // 3. Active Word Logic
  const safeWords = (words || []) as Word[];
  const activeToken = safeWords.find(w => currentTime >= (w.t0 ?? 0) && currentTime <= (w.t1 ?? 0));
  const t = activeToken ? (frame - ((activeToken.t0 ?? 0) * fps)) : 0;

  // 4. Layout Calculation
  const containerStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    position: 'absolute',
    top: 0,
    left: 0
  };

  if (manifest.layout.mode === 'top') {
    containerStyle.alignItems = 'flex-start';
    containerStyle.paddingTop = '15%';
  } else if (manifest.layout.mode === 'bottom') {
    containerStyle.alignItems = 'flex-end';
    containerStyle.paddingBottom = '15%';
  }

  return (
    <AbsoluteFill className="bg-black font-sans">
      <link rel="stylesheet" href={fontUrl} />
      {/* Background Layer: Reacts to audio energy (Pulse) */}
      <AbsoluteFill style={{
        transform: `scale(${1 + (smoothEnergy * (manifest.kinetics.reactivity || 1.0) * 0.03)})`,
        filter: manifest.kinetics.effect === 'glitch' && smoothEnergy > 0.8 ? 'hue-rotate(90deg) saturate(150%)' : 'none',
        transition: 'transform 0.1s ease-out'
      }}>
        {videoBgUrl ? (
          <OffthreadVideo src={videoBgUrl} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full bg-gray-900" />
        )}
      </AbsoluteFill>

      <Audio src={audioUrl} startFrom={(hook?.start || 0) * fps} />

      {/* Global Motion Layer */}
      <AbsoluteFill style={containerStyle}>
        {activeToken && (
          <KineticTypography
            text={activeToken.tw || activeToken.w}
            manifest={manifest}
            energy={smoothEnergy}
            t={t}
            fps={fps}
          />
        )}
      </AbsoluteFill>

      {/* Main Title Overlay (0-3s) */}
      {frame < 3 * fps && videoTitle && (
        <AbsoluteFill style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: 'rgba(0,0,0,0.3)',
        }}>
          <div style={{
            fontFamily: titleFontFamily || 'Syne',
            fontSize: '12rem',
            fontWeight: 900,
            color: 'white',
            textAlign: 'center',
            textTransform: 'uppercase',
            letterSpacing: '0.2em',
            opacity: interpolate(frame, [0, 10, (3 * fps) - 10, 3 * fps], [0, 1, 1, 0]),
            transform: `scale(${interpolate(frame, [0, 3 * fps], [0.9, 1.1])})`,
          }}>
            {videoTitle}
          </div>
        </AbsoluteFill>
      )}

      {/* Subtitles / Translations (Only if it's the secondary language) */}
      {activeToken?.tw && activeToken?.w && (
        <div style={{
          position: 'absolute',
          bottom: 100,
          width: '100%',
          textAlign: 'center',
          fontFamily: 'Inter',
          fontSize: 40,
          color: 'white',
          opacity: 0.4
        }}>
          {activeToken.w}
        </div>
      )}
    </AbsoluteFill>
  );
};
