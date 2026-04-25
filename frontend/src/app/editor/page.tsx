"use client";

import { useState, useCallback, useRef } from "react";
import EditorCanvas from "@/components/EditorCanvas";
import InventorySidebar from "@/components/InventorySidebar";
import FabricSidebar from "@/components/FabricSidebar";
import SettingsPanel from "@/components/SettingsPanel";
import Link from "next/link";
import { ArrowLeft, Sparkles, Settings, Download, Loader2, X, Sofa, Droplets } from "lucide-react";

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

interface FabricSwatch {
    id: string;
    name: string;
    thumbnail_url: string;
    full_url: string;
    base64?: string;
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
    const [selectedFabric, setSelectedFabric] = useState<FabricSwatch | null>(null);
    const [activeSidebar, setActiveSidebar] = useState<"inventory" | "fabric">("inventory");
    // Fabric polling state
    const [fabricJobId, setFabricJobId] = useState<string | null>(null);
    const [fabricStatus, setFabricStatus] = useState<string>(""); // human-readable progress
    const fabricPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // ── Callbacks ──

    const handleSegmentComplete = useCallback((result: SegmentResult) => {
        setClassLabel(result.class_label);
        setClassConfidence(result.class_confidence);
        setClassTop3(result.class_top3);
        setSelectedItem(null);
        setSelectedFabric(null);
        setPreviewImage(null);
    }, []);

    const handleImageLoaded = useCallback((loaded: boolean) => {
        setImageLoaded(loaded);
        if (!loaded) {
            setClassLabel(null);
            setClassConfidence(0);
            setClassTop3([]);
            setSelectedItem(null);
            setSelectedFabric(null);
            setPreviewImage(null);
        }
    }, []);

    const handleItemSelect = useCallback(async (item: InventoryItem | null) => {
        setSelectedItem(item);
        setPreviewImage(null);

        if (!item) return;

        // Generate live preview
        setIsGeneratingPreview(true);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

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
                signal: controller.signal,
            });

            if (res.ok) {
                const data = await res.json();
                setPreviewImage(data.preview_base64);
            } else {
                console.error("Preview generation failed:", res.status);
            }
        } catch (err) {
            if (err instanceof Error && err.name === "AbortError") {
                alert("Preview request timed out. This usually happens if your GPU is busy or models are still downloading. Please check the terminal.");
            } else {
                console.error("Preview error:", err);
            }
        } finally {
            clearTimeout(timeoutId);
            setIsGeneratingPreview(false);
        }
    }, []);

    // ── Helpers ──

    const stopFabricPoll = useCallback(() => {
        if (fabricPollRef.current !== null) {
            clearInterval(fabricPollRef.current);
            fabricPollRef.current = null;
        }
    }, []);

    const handleCancelFabric = useCallback(async () => {
        stopFabricPoll();
        try {
            await fetch("http://localhost:8000/api/cancel_fabric", { method: "POST" });
        } catch { /* ignore */ }
        setIsGeneratingPreview(false);
        setFabricJobId(null);
        setFabricStatus("");
    }, [stopFabricPoll]);

    const handleFabricSelect = useCallback(async (fabric: FabricSwatch | null) => {
        setSelectedFabric(fabric);
        setSelectedItem(null);
        setPreviewImage(null);

        if (!fabric) return;

        // Cancel any existing poll before starting a new job
        stopFabricPoll();
        await handleCancelFabric();

        setIsGeneratingPreview(true);
        setFabricStatus("Starting SDXL pipeline\u2026");

        try {
            const formData = new FormData();
            formData.append("fabric_url", fabric.full_url);

            const startRes = await fetch("http://localhost:8000/api/apply_fabric", {
                method: "POST",
                body: formData,
            });

            if (!startRes.ok) {
                const err = await startRes.json().catch(() => ({ detail: startRes.statusText }));
                throw new Error(err.detail || "Failed to start fabric job");
            }

            const startData = await startRes.json();
            const jobId: string = startData.job_id;
            setFabricJobId(jobId);
            setFabricStatus("Generating\u2026 (first run downloads ~16 GB of model weights)");

            // Poll every 5 s for completion
            let elapsedSeconds = 0;
            fabricPollRef.current = setInterval(async () => {
                elapsedSeconds += 5;

                try {
                    const pollRes = await fetch(
                        `http://localhost:8000/api/fabric_status/${jobId}`
                    );
                    const pollData = await pollRes.json();

                    if (pollData.status === "running") {
                        const mins = Math.floor(elapsedSeconds / 60);
                        const secs = elapsedSeconds % 60;
                        const elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                        setFabricStatus(`Generating\u2026 ${elapsed} elapsed (SDXL is working)`);
                        return; // keep polling
                    }

                    // Terminal state — stop polling
                    stopFabricPoll();
                    setFabricJobId(null);
                    setIsGeneratingPreview(false);

                    if (pollData.status === "done") {
                        setPreviewImage(pollData.result_base64);
                        setFabricStatus("");
                    } else if (pollData.status === "cancelled") {
                        setFabricStatus("Cancelled.");
                        setTimeout(() => setFabricStatus(""), 3000);
                    } else if (pollData.status === "error") {
                        console.error("Fabric job error:", pollData.detail);
                        alert(`Fabric application failed: ${pollData.detail}`);
                        setFabricStatus("");
                    } else if (pollData.status === "not_found") {
                        // Job was already cleaned up
                        setFabricStatus("");
                    }
                } catch (pollErr) {
                    console.error("Poll error:", pollErr);
                    // Network hiccup — keep polling
                }
            }, 5000);

        } catch (err) {
            stopFabricPoll();
            setIsGeneratingPreview(false);
            setFabricStatus("");
            console.error("Fabric apply error:", err);
            if (err instanceof Error) {
                alert(`Failed to apply fabric: ${err.message}`);
            }
        }
    }, [stopFabricPoll, handleCancelFabric]);

    const handleExport = useCallback(async () => {
        if (!selectedItem && !selectedFabric) {
            alert("Select a product or fabric first.");
            return;
        }

        setIsExporting(true);
        setExportProgress("Initializing SDXL pipeline...");

        try {
            let res;
            if (selectedFabric) {
                // Fabric application via direct blocking call (export button = user intent, no poll)
                setExportProgress("Applying fabric texture (may take 1\u20135 min)...");
                const formData = new FormData();
                formData.append("fabric_url", selectedFabric.full_url);
                formData.append("prompt", "");

                // Start the job
                const startRes = await fetch("http://localhost:8000/api/apply_fabric", {
                    method: "POST",
                    body: formData,
                });
                if (!startRes.ok) throw new Error(`Start failed: ${startRes.status}`);
                const { job_id } = await startRes.json();

                // Poll until done (no timeout — user explicitly clicked Export)
                let done = false;
                while (!done) {
                    await new Promise(r => setTimeout(r, 5000));
                    const pollRes = await fetch(`http://localhost:8000/api/fabric_status/${job_id}`);
                    const pollData = await pollRes.json();

                    if (pollData.status === "running") {
                        setExportProgress(`Applying fabric... (SDXL running)`);
                    } else if (pollData.status === "done") {
                        setPreviewImage(pollData.result_base64);
                        done = true;
                    } else {
                        throw new Error(pollData.detail || pollData.status);
                    }
                }
            } else if (selectedItem) {
                setExportProgress("Loading product image...");
                const imgRes = await fetch(selectedItem.full_url);
                const imgBlob = await imgRes.blob();

                const formData = new FormData();
                formData.append("product_image", imgBlob, "product.png");
                formData.append("category", classLabel || "");
                formData.append("prompt", "");

                setExportProgress("Running SDXL inpainting (may take 1\u20135 min on first run)...");

                res = await fetch("http://localhost:8000/api/export", {
                    method: "POST",
                    body: formData,
                });

                if (!res.ok) throw new Error(`Export failed with status ${res.status}`);
                const data = await res.json();
                setPreviewImage(data.export_base64 || data.result_base64);
            }

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
    }, [selectedItem, selectedFabric, classLabel]);

    return (
        <div className="h-screen w-full overflow-hidden bg-background flex flex-col">
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

                    {/* Preview loading indicator with cancel button */}
                        {isGeneratingPreview && (
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                <span className="max-w-[160px] truncate" title={fabricStatus || "Generating preview"}>
                                    {fabricStatus || "Generating preview"}
                                </span>
                                {fabricJobId && (
                                    <button
                                        onClick={handleCancelFabric}
                                        title="Cancel generation"
                                        className="ml-1 p-0.5 rounded hover:bg-blue-500/20 transition-colors"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                )}
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
                            disabled={!(selectedItem || selectedFabric) || isExporting}
                            className={`flex items-center gap-2 text-sm font-medium px-4 py-2 rounded-full transition-all ${
                                !(selectedItem || selectedFabric)
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

            {/* Sidebar Tabs & Workspace */}
            <div className="flex-1 flex overflow-hidden">
                {/* Sidebar Column */}
                <div className="w-80 border-r border-border/50 bg-card/30 flex flex-col shrink-0">
                    {/* Tab Switcher */}
                    <div className="flex p-2 gap-1 border-b border-white/5 bg-black/10">
                        <button
                            onClick={() => setActiveSidebar("inventory")}
                            className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all ${
                                activeSidebar === "inventory"
                                    ? "bg-white/10 text-white shadow-sm"
                                    : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                            }`}
                        >
                            <Sofa className="w-3.5 h-3.5" />
                            Inventory
                        </button>
                        <button
                            onClick={() => setActiveSidebar("fabric")}
                            className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all ${
                                activeSidebar === "fabric"
                                    ? "bg-white/10 text-white shadow-sm"
                                    : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                            }`}
                        >
                            <Droplets className="w-3.5 h-3.5" />
                            Fabric
                        </button>
                    </div>

                    {/* Active Sidebar Component */}
                    <div className="flex-1 overflow-hidden flex flex-col">
                        {activeSidebar === "inventory" ? (
                            <InventorySidebar
                                classLabel={classLabel}
                                classConfidence={classConfidence}
                                classTop3={classTop3}
                                onItemSelect={handleItemSelect}
                                selectedItem={selectedItem}
                                imageLoaded={imageLoaded}
                            />
                        ) : (
                            <FabricSidebar
                                onFabricSelect={handleFabricSelect}
                                selectedFabric={selectedFabric}
                                imageLoaded={imageLoaded}
                            />
                        )}
                    </div>
                </div>

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
