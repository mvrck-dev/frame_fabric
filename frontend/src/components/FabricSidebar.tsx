"use client";

import { useState } from "react";
import { Sparkles, Loader2, Wand2, Check } from "lucide-react";

interface FabricSwatch {
    id: string;
    name: string;
    thumbnail_url: string;
    full_url: string;
    base64?: string;
}

interface FabricSidebarProps {
    onFabricSelect: (fabric: FabricSwatch | null) => void;
    selectedFabric: FabricSwatch | null;
    imageLoaded: boolean;
}

export default function FabricSidebar({
    onFabricSelect,
    selectedFabric,
    imageLoaded,
}: FabricSidebarProps) {
    const [prompt, setPrompt] = useState("luxury velvet");
    const [fabrics, setFabrics] = useState<FabricSwatch[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);

    const handleGenerate = async () => {
        if (!prompt) return;
        setIsGenerating(true);
        try {
            const formData = new FormData();
            formData.append("prompt", prompt);
            formData.append("count", "4");

            const res = await fetch("http://localhost:8000/api/generate_fabrics", {
                method: "POST",
                body: formData,
            });

            if (res.ok) {
                const data = await res.json();
                setFabrics(data.fabrics || []);
            }
        } catch (err) {
            console.error("Failed to generate fabrics:", err);
        } finally {
            setIsGenerating(false);
        }
    };

    if (!imageLoaded) {
        return (
            <div className="flex-1 flex flex-col min-h-0">
                <div className="p-4 border-b border-border/50">
                    <h2 className="font-semibold text-sm">Fabric Generator</h2>
                    <p className="text-xs text-muted-foreground mt-1">
                        Select an object in the canvas first.
                    </p>
                </div>
                <div className="flex-1 p-4 flex flex-col items-center justify-center text-center text-muted-foreground/50">
                    <Sparkles className="w-8 h-8 mb-4 opacity-20" />
                    <p className="text-sm">Upload an image and select a piece of furniture to change its fabric.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col min-h-0">
            <div className="p-4 border-b border-border/50">
                <h2 className="font-semibold text-sm mb-4">Fabric Generator</h2>
                <div className="space-y-3">
                    <div className="relative">
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="e.g., luxury velvet, floral linen..."
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/40 transition-all"
                        />
                    </div>
                    <button
                        onClick={handleGenerate}
                        disabled={isGenerating || !prompt}
                        className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-xl py-2.5 text-sm font-medium transition-all shadow-lg shadow-purple-500/20"
                    >
                        {isGenerating ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Wand2 className="w-4 h-4" />
                        )}
                        Generate Fabrics
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
                {isGenerating && fabrics.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                        <Loader2 className="w-8 h-8 animate-spin text-purple-400 mb-4" />
                        <p className="text-xs">Synthesizing textures...</p>
                    </div>
                ) : fabrics.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
                        <div className="w-12 h-12 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-4">
                            <Sparkles className="w-5 h-5 opacity-20" />
                        </div>
                        <p className="text-sm font-medium text-foreground/80">No fabrics yet</p>
                        <p className="text-xs px-4 mt-1">Enter a prompt above to generate unique fabric textures with LoRA.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 gap-3">
                        {fabrics.map((fabric) => (
                            <button
                                key={fabric.id}
                                onClick={() => onFabricSelect(fabric)}
                                className={`relative aspect-square rounded-xl overflow-hidden border-2 transition-all hover:scale-[1.02] active:scale-[0.98] ${
                                    selectedFabric?.id === fabric.id
                                        ? "border-purple-500 ring-4 ring-purple-500/20"
                                        : "border-transparent hover:border-white/20"
                                }`}
                            >
                                <img
                                    src={fabric.thumbnail_url}
                                    alt={fabric.name}
                                    className="w-full h-full object-cover"
                                />
                                {selectedFabric?.id === fabric.id && (
                                    <div className="absolute inset-0 bg-purple-500/10 flex items-center justify-center">
                                        <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center shadow-lg">
                                            <Check className="w-5 h-5 text-white" />
                                        </div>
                                    </div>
                                )}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            <div className="p-4 border-t border-border/50 bg-card/50">
                <p className="text-[10px] text-muted-foreground text-center uppercase tracking-widest font-semibold">
                    LoRA Texture Synthesis
                </p>
            </div>
        </div>
    );
}
