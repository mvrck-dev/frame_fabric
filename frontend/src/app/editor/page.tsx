import EditorCanvas from "@/components/EditorCanvas";
import Link from "next/link";
import { ArrowLeft, Sparkles } from "lucide-react";

export default function EditorPage() {
    return (
        <div className="min-h-screen bg-background flex flex-col">
            {/* Top Navigation */}
            <nav className="border-b border-border/50 bg-background/50 backdrop-blur-xl shrink-0">
                <div className="flex h-16 items-center px-6 gap-4">
                    <Link
                        href="/"
                        className="p-2 -ml-2 hover:bg-white/5 rounded-full transition-colors text-muted-foreground hover:text-foreground"
                    >
                        <ArrowLeft className="w-5 h-5" />
                    </Link>
                    <div className="flex items-center gap-2 border-l border-border/50 pl-4">
                        <Sparkles className="w-5 h-5 text-purple-400" />
                        <span className="font-semibold tracking-tight">VisionPhase Editor</span>
                    </div>

                    <div className="ml-auto flex items-center gap-3">
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                            Engine Ready
                        </div>
                        <button className="text-sm font-medium bg-white text-black px-4 py-2 rounded-full hover:bg-gray-200 transition-colors">
                            Export Design
                        </button>
                    </div>
                </div>
            </nav>

            {/* Main Workspace */}
            <div className="flex-1 flex overflow-hidden">
                {/* Sidebar (Material Library Placeholder) */}
                <aside className="w-80 border-r border-border/50 bg-card/30 flex flex-col shrink-0">
                    <div className="p-4 border-b border-border/50">
                        <h2 className="font-semibold text-sm">Material Library</h2>
                        <p className="text-xs text-muted-foreground mt-1">Select an object in the canvas first.</p>
                    </div>
                    <div className="flex-1 p-4 flex flex-col items-center justify-center text-center text-muted-foreground/50">
                        <div className="w-16 h-16 rounded-full border-2 border-dashed border-border flex items-center justify-center mb-4">
                            <Sparkles className="w-6 h-6 opacity-30" />
                        </div>
                        <p className="text-sm px-4">Upload an image and tap on a surface to view available materials.</p>
                    </div>
                </aside>

                {/* Canvas Area */}
                <main className="flex-1 relative bg-black/40 flex items-center justify-center p-6 overflow-hidden">
                    <EditorCanvas />
                </main>
            </div>
        </div>
    );
}
