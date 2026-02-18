import React, { useMemo } from 'react';
import { useCurrentFrame, useVideoConfig, Audio, AbsoluteFill, OffthreadVideo } from 'remotion';
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
  const { fps, width, height } = useVideoConfig();
  const currentTime = frame / fps;

  // 1. Resolve Motion Manifest (Merge User Constraints if passed as loose props)
  const manifest: MotionManifest = useMemo(() => {
    // Default Fallback
    const base: MotionManifest = motion_manifest || {
      typography: { fontFamily: fontFamily || 'Inter', fontWeight: 800, fontSize: fontSize || 6, opacity: lyricOpacity ?? 1 },
      palette: { primary: lyricColor || '#ffffff', secondary: '#cccccc' },
      layout: { mode: (position as any) || 'center' },
      kinetics: { reactivity: 1.0, jitter: 0.0, effect: animationEffect || 'fade', physics: 'spring' }
    };

    // Ensure strict override if props exist (redundancy check)
    if (fontFamily) base.typography.fontFamily = fontFamily;
    if (fontSize) base.typography.fontSize = fontSize;
    if (lyricOpacity !== undefined) base.typography.opacity = lyricOpacity;
    if (position) base.layout.mode = position as any;
    if (animationEffect) base.kinetics.effect = animationEffect;

    return base;
  }, [motion_manifest, fontFamily, fontSize, lyricOpacity, position, animationEffect, lyricColor]);

  // 1a. Dynamic Font Loading
  // Load both main font and title font
  const fontsToLoad = Array.from(new Set([manifest.typography.fontFamily || 'Inter', titleFontFamily || 'Syne']));
  const fontUrl = `https://fonts.googleapis.com/css2?family=${fontsToLoad.map(f => f.replace(/\s+/g, '+')).join('&family=')}:wght@400;700;900&display=swap`;

  // 2. Audio Energy Calculation
  const currentSecond = Math.floor(currentTime);
  const localEnergy = energy_flux[currentSecond] || 0.5;
  const nextEnergy = energy_flux[currentSecond + 1] || localEnergy;
  const energyProgress = currentTime % 1;
  const smoothEnergy = localEnergy * (1 - energyProgress) + nextEnergy * energyProgress;

  // 3. Active Word Logic
  const activeToken = words.find(w => currentTime >= w.t0 && currentTime <= w.t1);
  const t = activeToken ? (frame - (activeToken.t0 * fps)) : 0;

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
      {/* Background Layer */}
      <AbsoluteFill>
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
