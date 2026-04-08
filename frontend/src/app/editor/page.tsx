"use client";

import { useState, useCallback, useRef } from "react";
import EditorCanvas from "@/components/EditorCanvas";
import InventorySidebar from "@/components/InventorySidebar";
import SettingsPanel from "@/components/SettingsPanel";
import Link from "next/link";
import { ArrowLeft, Sparkles, Settings, Download, Loader2, X } from "lucide-react";

interface SegmentResult {
    mask_base64: string;
    class_label: string;
    class_confidence: number;
    class_top3: { label: string; confidence: number }[];
}

interface InventoryItem {
    name: string;
    filename: string;
    thumbnail_url: string;
    full_url: string;
}

export default function EditorPage() {
    // ── State ──
    const [imageLoaded, setImageLoaded] = useState(false);
    const [classLabel, setClassLabel] = useState<string | null>(null);
    const [classConfidence, setClassConfidence] = useState(0);
    const [classTop3, setClassTop3] = useState<{ label: string; confidence: number }[]>([]);
    const [selectedItem, setSelectedItem] = useState<InventoryItem | null>(null);
    const [previewImage, setPreviewImage] = useState<string | null>(null);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState("");
    const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);

    // ── Callbacks ──

    const handleSegmentComplete = useCallback((result: SegmentResult) => {
        setClassLabel(result.class_label);
        setClassConfidence(result.class_confidence);
        setClassTop3(result.class_top3);
        setSelectedItem(null);
        setPreviewImage(null);
    }, []);

    const handleImageLoaded = useCallback((loaded: boolean) => {
        setImageLoaded(loaded);
        if (!loaded) {
            setClassLabel(null);
            setClassConfidence(0);
            setClassTop3([]);
            setSelectedItem(null);
            setPreviewImage(null);
        }
    }, []);

    const handleItemSelect = useCallback(async (item: InventoryItem | null) => {
        setSelectedItem(item);
        setPreviewImage(null);

        if (!item) return;

        // Generate live preview
        setIsGeneratingPreview(true);
        try {
            // Fetch the product image and send to preview endpoint
            const imgRes = await fetch(item.full_url);
            const imgBlob = await imgRes.blob();

            const formData = new FormData();
            formData.append("product_image", imgBlob, "product.png");
            formData.append("product_name", item.name);

            const res = await fetch("http://localhost:8000/api/preview", {
                method: "POST",
                body: formData,
            });

            if (res.ok) {
                const data = await res.json();
                setPreviewImage(data.preview_base64);
            } else {
                console.error("Preview generation failed:", res.status);
            }
        } catch (err) {
            console.error("Preview error:", err);
        } finally {
            setIsGeneratingPreview(false);
        }
    }, []);

    const handleExport = useCallback(async () => {
        if (!selectedItem) {
            alert("Select a product from the inventory first.");
            return;
        }

        setIsExporting(true);
        setExportProgress("Initializing SDXL pipeline...");

        try {
            // Fetch product image
            setExportProgress("Loading product image...");
            const imgRes = await fetch(selectedItem.full_url);
            const imgBlob = await imgRes.blob();

            const formData = new FormData();
            formData.append("product_image", imgBlob, "product.png");
            formData.append("category", classLabel || "");
            formData.append("prompt", "");

            setExportProgress("Running SDXL inpainting (this may take 30-60s)...");

            const res = await fetch("http://localhost:8000/api/export", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                throw new Error(`Export failed with status ${res.status}`);
            }

            const data = await res.json();

            // Display the high-quality exported image on the canvas instead of downloading
            setExportProgress("Finalizing render...");
            setPreviewImage(data.export_base64);

            setExportProgress("Render complete!");
            setTimeout(() => {
                setIsExporting(false);
                setExportProgress("");
            }, 2000);

        } catch (err) {
            console.error("Export error:", err);
            setExportProgress("Export failed. Check console for details.");
            setTimeout(() => {
                setIsExporting(false);
                setExportProgress("");
            }, 3000);
        }
    }, [selectedItem, classLabel]);

    return (
        <div className="min-h-screen bg-background flex flex-col">
            {/* Top Navigation */}
            <nav className="border-b border-border/50 bg-background/50 backdrop-blur-xl shrink-0 z-30">
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
                        {/* Engine Status */}
                        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                            Engine Ready
                        </div>

                        {/* Preview loading indicator */}
                        {isGeneratingPreview && (
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Preview
                            </div>
                        )}

                        {/* Settings Button */}
                        <button
                            onClick={() => setSettingsOpen(true)}
                            className="p-2 rounded-full hover:bg-white/5 text-muted-foreground hover:text-foreground transition-colors border border-white/8"
                            title="Pipeline Settings"
                        >
                            <Settings className="w-4 h-4" />
                        </button>

                        {/* Export Button */}
                        <button
                            onClick={handleExport}
                            disabled={!selectedItem || isExporting}
                            className={`flex items-center gap-2 text-sm font-medium px-4 py-2 rounded-full transition-all ${
                                !selectedItem
                                    ? "bg-white/5 text-muted-foreground cursor-not-allowed border border-white/8"
                                    : isExporting
                                        ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                                        : "bg-white text-black hover:bg-gray-200 hover:scale-105 active:scale-95"
                            }`}
                        >
                            {isExporting ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    Exporting...
                                </>
                            ) : (
                                <>
                                    <Download className="w-4 h-4" />
                                    Export Design
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </nav>

            {/* Main Workspace */}
            <div className="flex-1 flex overflow-hidden">
                {/* Sidebar — Inventory */}
                <InventorySidebar
                    classLabel={classLabel}
                    classConfidence={classConfidence}
                    classTop3={classTop3}
                    onItemSelect={handleItemSelect}
                    selectedItem={selectedItem}
                    imageLoaded={imageLoaded}
                />

                {/* Canvas Area */}
                <main className="flex-1 relative bg-black/40 flex items-center justify-center p-6 overflow-hidden">
                    <EditorCanvas
                        onSegmentComplete={handleSegmentComplete}
                        onImageLoaded={handleImageLoaded}
                        previewImage={previewImage}
                    />
                </main>
            </div>

            {/* Settings Panel */}
            <SettingsPanel
                isOpen={settingsOpen}
                onClose={() => setSettingsOpen(false)}
            />

            {/* Export Progress Modal */}
            {isExporting && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center">
                    <div className="glass-card rounded-2xl p-8 max-w-md w-full mx-4 border border-white/10">
                        <div className="flex flex-col items-center text-center">
                            <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mb-6 border border-purple-500/30">
                                <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
                            </div>
                            <h3 className="text-lg font-semibold mb-2">Exporting Design</h3>
                            <p className="text-sm text-muted-foreground mb-4">
                                {exportProgress}
                            </p>
                            <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                                <div className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full animate-pulse" style={{ width: "60%" }} />
                            </div>
                            <p className="text-xs text-muted-foreground mt-3">
                                First export downloads ~9 GB of model weights.
                                Subsequent exports are much faster.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
