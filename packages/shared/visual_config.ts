export type AnimationType =
    | 'spring-scale'
    | 'slide-up'
    | 'fade-in'
    | 'karaoke-highlight'
    | 'kinetic-type'
    | 'glitch'
    | 'neon'
    | 'bounce'
    | 'rotate'
    | 'blur-reveal'
    | 'magnetic'
    | 'float'
    | 'ghost'
    | 'pixel'
    | 'staggered';

export interface VisualPreset {
    id: string;
    name: string;
    fontFamily: string;
    animation: AnimationType;
    colors: string[]; // Gradients
    tags: string[]; // For AI mapping
}

export const VISUAL_PRESETS: VisualPreset[] = [
    {
        id: 'cyberflow',
        name: 'Cyber Flow',
        fontFamily: 'JetBrains Mono',
        animation: 'glitch',
        colors: ['#00ffff', '#ff00ff'],
        tags: ['cyberpunk', 'tech', 'digital', 'fast', 'energetic']
    },
    {
        id: 'editorial',
        name: 'Editorial Elegance',
        fontFamily: 'Playfair Display',
        animation: 'fade-in',
        colors: ['#ffffff', '#f0f0f0'],
        tags: ['minimalist', 'clean', 'elegant', 'fashion', 'sad', 'calm']
    },
    {
        id: 'tiktok_bold',
        name: 'TikTok Bold',
        fontFamily: 'Anton',
        animation: 'spring-scale',
        colors: ['#ffff00', '#ff0000'],
        tags: ['aggressive', 'bold', 'loud', 'viral', 'heavy']
    },
    {
        id: 'soft_glow',
        name: 'Soft Glow',
        fontFamily: 'Outfit',
        animation: 'neon',
        colors: ['#ff99cc', '#cc99ff'],
        tags: ['dreamy', 'pop', 'soft', 'love', 'happy']
    },
    {
        id: 'bouncy_pop',
        name: 'Bouncy Pop',
        fontFamily: 'Bungee',
        animation: 'bounce',
        colors: ['#ff6600', '#ffff00'],
        tags: ['fun', 'playful', 'childish', 'happy', 'energetic']
    },
    {
        id: 'retro_arcade',
        name: 'Retro Arcade',
        fontFamily: 'Unbounded',
        animation: 'pixel',
        colors: ['#00ff00', '#000000'],
        tags: ['retro', 'gaming', '8bit', 'old-school']
    },
    {
        id: 'float_dream',
        name: 'Floating Dream',
        fontFamily: 'Syne',
        animation: 'float',
        colors: ['#66ffff', '#3366ff'],
        tags: ['ambient', 'chill', 'electronic', 'dreamy']
    },
    {
        id: 'urban_grit',
        name: 'Urban Grit',
        fontFamily: 'Archivo Black',
        animation: 'slide-up',
        colors: ['#cccccc', '#333333'],
        tags: ['street', 'raw', 'rap', 'hiphop', 'dark']
    },
    {
        id: 'kinetic_pro',
        name: 'Kinetic Pro',
        fontFamily: 'Inter',
        animation: 'kinetic-type',
        colors: ['#ffffff', '#999999'],
        tags: ['modern', 'corporate', 'explain', 'sharp']
    },
    {
        id: 'liquid_vibe',
        name: 'Liquid Vibe',
        fontFamily: 'Space Grotesk',
        animation: 'blur-reveal',
        colors: ['#0099ff', '#6600ff'],
        tags: ['fluid', 'wavvy', 'psychedelic', 'art']
    },
    {
        id: 'ghost_notes',
        name: 'Ghost Notes',
        fontFamily: 'Fraunces',
        animation: 'ghost',
        colors: ['#ffffff', '#888888'],
        tags: ['mysterious', 'noir', 'jazz', 'smoke']
    },
    {
        id: 'karaoke_pro',
        name: 'Karaoke Pro',
        fontFamily: 'Lexend',
        animation: 'karaoke-highlight',
        colors: ['#ffcc00', '#ffffff'],
        tags: ['lyric', 'sing-along', 'classic', 'clean']
    },
    {
        id: 'magnetic_attract',
        name: 'Magnetic Attraction',
        fontFamily: 'Kanit',
        animation: 'magnetic',
        colors: ['#ff0066', '#ffcc00'],
        tags: ['intense', 'focus', 'center', 'hot']
    },
    {
        id: 'stagger_reveal',
        name: 'Stagger Reveal',
        fontFamily: 'Montserrat',
        animation: 'staggered',
        colors: ['#ffffff', '#00ccff'],
        tags: ['reveal', 'impact', 'cinematic']
    },
    {
        id: 'impact_now',
        name: 'Impact Now',
        fontFamily: 'Bebas Neue',
        animation: 'spring-scale',
        colors: ['#ff0000', '#ffffff'],
        tags: ['urgent', 'news', 'breaking', 'viral']
    }
];
