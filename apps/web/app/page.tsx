import { CreativeStudio } from "@/components/creative-studio";
import { Sparkles, Zap, ShieldCheck, Globe, Play, Cpu, Layers } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#030303] text-white selection:bg-indigo-500/30 font-sans mesh-gradient">
      {/* Background Cinematic Atmos */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-indigo-600/5 rounded-full blur-[150px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-purple-600/5 rounded-full blur-[150px] animate-pulse delay-1000" />

        {/* Animated Particles Simulation (Native CSS) */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/3 w-1 h-1 bg-white rounded-full animate-float" style={{ animationDelay: '1s' }} />
          <div className="absolute top-1/2 left-1/2 w-1.5 h-1.5 bg-indigo-400 rounded-full animate-float" style={{ animationDelay: '3s' }} />
          <div className="absolute top-2/3 left-1/4 w-1 h-1 bg-purple-400 rounded-full animate-float" style={{ animationDelay: '5s' }} />
        </div>
      </div>

      <div className="relative z-10 px-6 py-12 lg:px-12">
        {/* Premium Header */}
        <header className="max-w-[1500px] mx-auto flex flex-col items-center text-center gap-8 mb-24 relative z-20">

          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass-card text-[10px] font-black uppercase tracking-[0.3em] text-indigo-400">
            <Sparkles className="w-3.5 h-3.5 animate-pulse" />
            Multimodal Synesthesia v3.1
          </div>

          <div className="relative inline-block">
            <h1 className="text-7xl md:text-9xl font-black tracking-tighter leading-[0.85] text-glow">
              TOHJO<br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-200 to-indigo-500 italic font-light tracking-widest">STUDIO</span>
            </h1>
          </div>

          <p className="text-white/50 font-medium max-w-xl text-lg lg:text-xl leading-relaxed">
            Transcending the boundaries of audio. Our **Neural Orchestration** engine transforms lyrics into cinematic reality using **Local LTX-Video**.
          </p>
        </header>

        {/* Main Interface: Creative Studio */}
        <section className="relative">
          <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-b from-transparent via-white/10 to-transparent" />
          <CreativeStudio />
        </section>

        {/* Cinematic Footer */}
        <footer className="max-w-[1500px] mx-auto mt-40 border-t border-white/5 pt-16 pb-12 flex flex-col md:flex-row items-center justify-between gap-12 text-white/10">
          <div className="flex items-center gap-6">
            <Zap className="w-6 h-6 text-indigo-500 animate-pulse" />
            <div className="flex flex-col">
              <span className="text-[10px] font-black uppercase tracking-[0.4em]">Proprietary AI Stack</span>
              <span className="text-[9px] uppercase tracking-widest text-white/5">Distributed Node Network Alpha</span>
            </div>
          </div>

          <div className="flex flex-wrap justify-center gap-x-12 gap-y-4 font-black text-[10px] uppercase tracking-[0.2em] grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-700">
            <span className="hover:text-indigo-400 cursor-default">Fastify v4</span>
            <span className="hover:text-purple-400 cursor-default">Local LTX-Video</span>
            <span className="hover:text-blue-400 cursor-default">Supabase Orion</span>
            <span className="hover:text-pink-400 cursor-default">Remotion Render</span>
            <span className="hover:text-green-400 cursor-default">BullMQ Flow</span>
          </div>
        </footer>
      </div>
    </main>
  );
}
