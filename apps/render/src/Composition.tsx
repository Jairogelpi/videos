import React from 'react';
import { useCurrentFrame, useVideoConfig, Audio, AbsoluteFill, interpolate, spring, Easing, OffthreadVideo } from 'remotion';
import { VISUAL_PRESETS } from '@repo/shared';

interface Word {
  t0: number;
  t1: number;
  w: string;
  conf?: number;
  lang?: string;
  tw?: string; // Translated word/text
  type?: 'lyric' | 'phrase' | 'caption';
}

interface Props {
  words: Word[];
  audioUrl: string;
  styleId?: string;
  mood?: string;
  position?: 'top' | 'center' | 'bottom';
  fontSize?: number;
  videoBgUrl?: string;
  hook?: { start: number; end: number };
  energy_flux?: number[];
  harmonic_key?: string;
  vocal_brightness?: number;
  fontFamily?: string;
  lyricColor?: string;
  lyricOpacity?: number;
  animationEffect?: string;
}

const SCALING_FACTOR = 3.0; // Ratio 1080 / 360

export const MyComposition: React.FC<Props> = ({
  words = [],
  audioUrl,
  styleId = 'tiktok_bold',
  mood = 'default',
  position = 'center',
  fontSize = 6,
  videoBgUrl,
  hook,
  energy_flux = [],
  harmonic_key = 'C Major',
  vocal_brightness = 2000,
  fontFamily,
  lyricColor,
  lyricOpacity = 1,
  animationEffect = 'pop'
}) => {
  const frame = useCurrentFrame();
  const { fps, height } = useVideoConfig();
  const currentTime = frame / fps;
  const relativeTime = currentTime;

  // 1. NEURAL SYNESTHESIA: Energy Reactivity
  const currentSecond = Math.floor(relativeTime);
  const localEnergy = energy_flux[currentSecond] || 0.5;
  const nextEnergy = energy_flux[currentSecond + 1] || localEnergy;
  const energyProgress = relativeTime % 1;
  const smoothEnergy = localEnergy * (1 - energyProgress) + nextEnergy * energyProgress;

  // Kinetic Typography FX
  const reactiveScale = 1 + (smoothEnergy * 0.1);
  const reactiveBlur = Math.max(0, (1 - smoothEnergy) * 3); // More blur on low energy

  // Find active word or phrase
  const activeToken = words.find(w => relativeTime >= w.t0 && relativeTime <= w.t1);

  // Resolve Style Preset
  const style = VISUAL_PRESETS.find((p: any) => p.id === styleId) || VISUAL_PRESETS[0];
  const finalEffect = animationEffect || style.animation;

  // Entrance Timing (Normalized to frame 0 of word appearance)
  const t = activeToken ? (frame - (activeToken.t0 * fps)) : 0;

  // --- ANIMATION SUITE ---

  // POP / SPRING
  const popScale = spring({
    frame: t,
    fps,
    config: { stiffness: 200, damping: 12 }
  });

  // FADE
  const opacityFade = interpolate(t, [0, 5], [0, 1], { extrapolateRight: 'clamp' });

  // SLIDE
  const slideY = interpolate(t, [0, 6], [100 * SCALING_FACTOR, 0], {
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.quad)
  });

  // SHAKE (Deterministic jitter)
  const shakeX = (Math.sin(frame * 0.5) * 10 * SCALING_FACTOR) * (localEnergy > 0.8 ? 1 : 0);
  const shakeY = (Math.cos(frame * 0.7) * 10 * SCALING_FACTOR) * (localEnergy > 0.8 ? 1 : 0);

  // NEON PULSE
  const neonPulse = 0.9 + Math.sin(frame / 5) * 0.1;

  // POSITIONING
  const positionOffsets = {
    top: height * 0.15,
    center: height * 0.5,
    bottom: height * 0.85
  };
  const targetY = positionOffsets[position || 'center'];

  return (
    <AbsoluteFill className="bg-black overflow-hidden font-sans">
      {/* Background Layer: OffthreadVideo for zero flickering */}
      <AbsoluteFill>
        {videoBgUrl ? (
          <OffthreadVideo
            src={videoBgUrl}
            className="w-full h-full object-cover"
          />
        ) : (
          <div
            className="w-full h-full"
            style={{
              background: `radial-gradient(circle at center, ${lyricColor || style.colors[0]}26 0%, #000000 100%)`,
              opacity: 1
            }}
          />
        )}
      </AbsoluteFill>

      <Audio src={audioUrl} startFrom={(hook?.start || 0) * fps} />

      {/* Kinetic Typography Layer */}
      <AbsoluteFill className="flex items-center justify-center">
        {activeToken && (
          <div
            className="text-center absolute w-full px-32"
            style={{
              top: targetY,
              transform: `translateY(-50%)`,
            }}
          >
            <div
              style={{
                fontFamily: fontFamily || style.fontFamily,
                color: lyricColor || style.colors[0],
                fontSize: `${fontSize || 6}rem`,
                fontWeight: 900,
                textTransform: 'uppercase',
                display: 'inline-block',
                lineHeight: 0.85,
                filter: `blur(${reactiveBlur}px) drop-shadow(0 0 ${20 * SCALING_FACTOR}px ${(lyricColor || style.colors[0])}66)`,
                opacity: (lyricOpacity ?? 1) * (finalEffect === 'fade' ? opacityFade : 1) * (finalEffect === 'neon' ? neonPulse : 1),
                transform: `
                  translateY(${finalEffect === 'slide' ? slideY : 0}px)
                  translate(${finalEffect === 'shake' ? shakeX : 0}px, ${finalEffect === 'shake' ? shakeY : 0}px)
                  scale(${finalEffect === 'pop' || finalEffect === 'spring-scale' ? popScale * reactiveScale : reactiveScale})
                `
              }}
            >
              {activeToken.w}

              {/* Translation (Captions) */}
              {activeToken.tw && (
                <div
                  className="mt-8 opacity-60 font-medium lowercase"
                  style={{
                    fontFamily: 'Inter',
                    fontSize: '0.4em',
                    color: '#ffffff',
                    letterSpacing: '-0.02em'
                  }}
                >
                  {activeToken.tw}
                </div>
              )}
            </div>
          </div>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
