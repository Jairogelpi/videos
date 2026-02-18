export interface MotionManifest {
    typography: {
        fontFamily: string;
        fontWeight: number;
        tracking?: number; // letter-spacing in em
        fontSize?: number; // scaler for base size
        perCharacter?: boolean; // animate characters individually
        opacity?: number;
    };
    palette: {
        primary: string; // Hex
        secondary: string; // Hex
        accent?: string;
        shadow?: string;
        glow?: string;
    };
    layout: {
        mode: 'center' | 'top' | 'bottom' | 'scatter';
        offsetY?: number; // +/- percentage
        maxWidth?: string; // e.g. "80%"
    };
    kinetics: {
        effect?: string; // 'fade', 'pop', 'slide', 'typewriter', 'neon', 'glitch'
        reactivity: number; // 0.0 to 2.0 (Energy multiplier)
        jitter: number; // 0.0 to 1.0 (Shake intensity)
        flash?: boolean; // Flash on beat
        physics: 'spring' | 'linear' | 'elastic' | 'bounce';
    };
}
