import Link from "next/link";
import { ArrowRight, Sparkles, Wand2, Layers } from "lucide-react";

export default function Home() {
  return (
    <div className="relative min-h-screen flex flex-col items-center overflow-hidden bg-background">
      {/* Background Gradients */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-purple-900/20 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-blue-900/20 blur-[120px] pointer-events-none" />

      {/* Navigation */}
      <nav className="w-full flex justify-between items-center py-6 px-8 max-w-7xl relative z-10">
        <div className="flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-purple-400" />
          <span className="text-xl font-bold tracking-tight">VisionPhase</span>
        </div>
        <div className="flex items-center gap-6 text-sm font-medium text-muted-foreground">
          <Link href="#features" className="hover:text-foreground transition-colors">Features</Link>
          <Link href="#how-it-works" className="hover:text-foreground transition-colors">How it works</Link>
          <Link
            href="/editor"
            className="text-foreground bg-white/5 hover:bg-white/10 px-4 py-2 rounded-full border border-white/10 transition-all"
          >
            Launch Editor
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center text-center px-6 mt-16 relative z-10 max-w-5xl w-full">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-purple-300 mb-8 animate-fade-in">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
          </span>
          Neural Interior Redesign Engine v1.0
        </div>

        <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-6 leading-tight">
          Redesign your space with <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-500">
            zero 3D modeling.
          </span>
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mb-10 leading-relaxed">
          Upload a photo of your room. Tap any object or surface. Let our hybrid Diffusion-GAN architecture synthesize photorealistic replacements instantly.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-4">
          <Link
            href="/editor"
            className="group flex items-center gap-2 bg-foreground text-background px-8 py-4 rounded-full font-semibold hover:bg-gray-200 transition-all hover:scale-105 active:scale-95"
          >
            Start Editing
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>

          <Link
            href="#demo"
            className="flex items-center gap-2 px-8 py-4 rounded-full font-semibold border border-white/10 hover:bg-white/5 transition-all"
          >
            Watch Demo
          </Link>
        </div>

        {/* Features Preview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-24 mb-16 w-full text-left">
          <div className="glass-card p-6 rounded-2xl flex flex-col gap-4">
            <div className="h-12 w-12 rounded-full bg-purple-500/20 flex items-center justify-center border border-purple-500/30">
              <Layers className="text-purple-400 w-6 h-6" />
            </div>
            <h3 className="font-semibold text-lg">Zero-Shot Masking</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Powered by Meta's SAM. Tap anywhere to isolate walls, floors, or furniture with pixel-perfect accuracy.
            </p>
          </div>

          <div className="glass-card p-6 rounded-2xl flex flex-col gap-4">
            <div className="h-12 w-12 rounded-full bg-blue-500/20 flex items-center justify-center border border-blue-500/30">
              <Sparkles className="text-blue-400 w-6 h-6" />
            </div>
            <h3 className="font-semibold text-lg">Diffusion + GAN</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Stable Diffusion creates the concept. Our custom SPADE-GAN perfects the high-frequency textures and lighting.
            </p>
          </div>

          <div className="glass-card p-6 rounded-2xl flex flex-col gap-4">
            <div className="h-12 w-12 rounded-full bg-emerald-500/20 flex items-center justify-center border border-emerald-500/30">
              <Wand2 className="text-emerald-400 w-6 h-6" />
            </div>
            <h3 className="font-semibold text-lg">Lighting Preservation</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Existing shadows, global illumination, and perspective parameters are strictly maintained using geometric conditioning.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
