"use client";

import { useState, useEffect, useRef } from "react";
import { Sparkles, Upload, Loader2, Tag, ChevronDown } from "lucide-react";

interface InventoryItem {
    name: string;
    filename: string;
    thumbnail_url: string;
    full_url: string;
}

interface InventorySidebarProps {
    classLabel: string | null;
    classConfidence: number;
    classTop3: { label: string; confidence: number }[];
    onItemSelect: (item: InventoryItem | null) => void;
    selectedItem: InventoryItem | null;
    imageLoaded: boolean;
}

export default function InventorySidebar({
    classLabel,
    classConfidence,
    classTop3,
    onItemSelect,
    selectedItem,
    imageLoaded,
}: InventorySidebarProps) {
    const [items, setItems] = useState<InventoryItem[]>([]);
    const [category, setCategory] = useState<string>("");
    const [isLoading, setIsLoading] = useState(false);
    const [showClassDropdown, setShowClassDropdown] = useState(false);
    const [activeClassLabel, setActiveClassLabel] = useState<string | null>(null);
    const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
    const customUploadRef = useRef<HTMLInputElement>(null);

    // When classLabel changes, update and fetch inventory
    useEffect(() => {
        if (classLabel) {
            setActiveClassLabel(classLabel);
        }
    }, [classLabel]);

    // Fetch inventory when active class changes
    useEffect(() => {
        if (!activeClassLabel) return;

        const fetchInventory = async () => {
            setIsLoading(true);
            try {
                const res = await fetch(
                    `http://localhost:8000/api/inventory/${activeClassLabel}`
                );
                const data = await res.json();
                setItems(data.items || []);
                setCategory(data.category || "");
            } catch (err) {
                console.error("Failed to fetch inventory:", err);
                setItems([]);
            } finally {
                setIsLoading(false);
            }
        };

        fetchInventory();
    }, [activeClassLabel]);

    const handleSelectItem = async (item: InventoryItem) => {
        onItemSelect(item);
    };

    const handleCustomUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("http://localhost:8000/api/inventory/custom", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            if (data.status === "success") {
                const customItem: InventoryItem = {
                    name: data.name,
                    filename: "",
                    thumbnail_url: data.thumbnail_url,
                    full_url: data.full_url,
                };
                setItems(prev => [...prev, customItem]);
                onItemSelect(customItem);
            }
        } catch (err) {
            console.error("Custom upload failed:", err);
        }
    };

    // ── Empty state: no image loaded yet ──
    if (!imageLoaded) {
        return (
            <div className="flex-1 flex flex-col min-h-0">
                <div className="p-4 border-b border-border/50">
                    <h2 className="font-semibold text-sm">Material Library</h2>
                    <p className="text-xs text-muted-foreground mt-1">
                        Select an object in the canvas first.
                    </p>
                </div>
                <div className="flex-1 p-4 flex flex-col items-center justify-center text-center text-muted-foreground/50">
                    <div className="w-16 h-16 rounded-full border-2 border-dashed border-border flex items-center justify-center mb-4">
                        <Sparkles className="w-6 h-6 opacity-30" />
                    </div>
                    <p className="text-sm px-4">
                        Upload an image and tap on a surface to view available materials.
                    </p>
                </div>
            </div>
        );
    }

    // ── Waiting for first click ──
    if (!classLabel && !activeClassLabel) {
        return (
            <div className="flex-1 flex flex-col min-h-0">
                <div className="p-4 border-b border-border/50">
                    <h2 className="font-semibold text-sm">Material Library</h2>
                    <p className="text-xs text-muted-foreground mt-1">
                        Click on an object in the image to identify it.
                    </p>
                </div>
                <div className="flex-1 p-4 flex flex-col items-center justify-center text-center text-muted-foreground/50">
                    <div className="w-16 h-16 rounded-full border-2 border-dashed border-border flex items-center justify-center mb-4 animate-pulse">
                        <Tag className="w-6 h-6 opacity-30" />
                    </div>
                    <p className="text-sm px-4">
                        Tap any object in the scene to detect its type and browse matching materials.
                    </p>
                </div>
            </div>
        );
    }

    // ── Active state: show inventory ──
    return (
        <div className="flex-1 flex flex-col min-h-0">
            {/* Header with class label */}
            <div className="p-4 border-b border-border/50">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="font-semibold text-sm">Material Library</h2>
                    {category && (
                        <span className="text-xs text-muted-foreground tracking-wide uppercase">
                            {category}
                        </span>
                    )}
                </div>

                {/* Detected class badge */}
                <div className="relative">
                    <button
                        onClick={() => setShowClassDropdown(!showClassDropdown)}
                        className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-300 text-sm hover:bg-purple-500/15 transition-colors"
                    >
                        <div className="flex items-center gap-2">
                            <Tag className="w-3.5 h-3.5" />
                            <span className="font-medium capitalize">{activeClassLabel}</span>
                            <span className="text-purple-400/60 text-xs">
                                {Math.round(classConfidence * 100)}%
                            </span>
                        </div>
                        <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showClassDropdown ? "rotate-180" : ""}`} />
                    </button>

                    {/* Dropdown: alternative class labels */}
                    {showClassDropdown && classTop3.length > 0 && (
                        <div className="absolute top-full left-0 right-0 mt-1 glass-card rounded-lg border border-white/10 overflow-hidden z-50">
                            {classTop3.map(({ label, confidence }) => (
                                <button
                                    key={label}
                                    onClick={() => {
                                        setActiveClassLabel(label);
                                        setShowClassDropdown(false);
                                    }}
                                    className={`w-full px-3 py-2 text-left text-sm flex justify-between items-center hover:bg-white/5 transition-colors ${
                                        label === activeClassLabel ? "text-purple-400 bg-white/5" : "text-muted-foreground"
                                    }`}
                                >
                                    <span className="capitalize">{label}</span>
                                    <span className="text-xs opacity-60">
                                        {Math.round(confidence * 100)}%
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Inventory grid */}
            {activeClassLabel && (
                activeClassLabel.toLowerCase().includes("wall") || 
                activeClassLabel.toLowerCase().includes("ceiling") || 
                activeClassLabel.toLowerCase().includes("floor")
            ) ? (
                <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center justify-center text-center text-muted-foreground/50">
                    <div className="w-16 h-16 rounded-full border-2 border-dashed border-red-500/30 flex items-center justify-center mb-4 text-red-500/50 hover:bg-red-500/5 transition-colors">
                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                    </div>
                    <p className="text-sm font-medium text-foreground mb-2">Structural Element Detected</p>
                    <p className="text-xs px-2">
                        {activeClassLabel.charAt(0).toUpperCase() + activeClassLabel.slice(1)}s cannot be replaced with standalone furniture. Please select a different object.
                    </p>
                </div>
            ) : (
                <div className="flex-1 overflow-y-auto p-3">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                        <Loader2 className="w-8 h-8 animate-spin text-purple-400 mb-3" />
                        <span className="text-xs">Loading inventory...</span>
                    </div>
                ) : items.length === 0 ? (
                    <div className="flex-1 p-4 flex flex-col items-center justify-center text-center text-muted-foreground">
                        <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center mb-4">
                            <Tag className="w-5 h-5 opacity-20" />
                        </div>
                        <p className="text-sm font-medium text-foreground mb-2">No Matches Found</p>
                        <p className="text-xs px-4">
                            We don&apos;t have any furniture categories that match &ldquo;{activeClassLabel}&rdquo; yet. 
                            Try uploading a custom product image below.
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 gap-2">
                        {items.map((item, idx) => (
                            <button
                                key={`${item.filename}-${idx}`}
                                onClick={() => handleSelectItem(item)}
                                className={`relative group rounded-xl overflow-hidden border transition-all hover:scale-[1.02] active:scale-[0.98] ${
                                    selectedItem?.filename === item.filename && selectedItem?.name === item.name
                                        ? "border-purple-500 ring-2 ring-purple-500/30 shadow-lg shadow-purple-500/10"
                                        : "border-white/8 hover:border-white/20"
                                }`}
                            >
                                <div className="aspect-square bg-black/30 relative overflow-hidden">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                        src={item.thumbnail_url}
                                        alt={item.name}
                                        className="w-full h-full object-cover transition-transform group-hover:scale-110"
                                    />
                                    {/* Hover overlay */}
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                </div>
                                <div className="px-2 py-1.5 bg-card/80 border-t border-white/5">
                                    <p className="text-xs font-medium truncate">{item.name}</p>
                                </div>

                                {/* Selected checkmark */}
                                {selectedItem?.filename === item.filename && selectedItem?.name === item.name && (
                                    <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-purple-500 flex items-center justify-center shadow-lg">
                                        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                        </svg>
                                    </div>
                                )}
                            </button>
                        ))}
                    </div>
                )}
            </div>
            )}

            {/* Custom upload button */}
            {!(activeClassLabel && (
                activeClassLabel.toLowerCase().includes("wall") || 
                activeClassLabel.toLowerCase().includes("ceiling") || 
                activeClassLabel.toLowerCase().includes("floor")
            )) && (
                <div className="p-3 border-t border-border/50">
                    <button
                        onClick={() => customUploadRef.current?.click()}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-dashed border-white/15 text-sm font-medium text-muted-foreground hover:text-foreground hover:border-purple-500/40 hover:bg-purple-500/5 transition-all"
                    >
                        <Upload className="w-4 h-4" />
                        Upload Custom Product
                    </button>
                    <input
                        type="file"
                        className="hidden"
                        ref={customUploadRef}
                        accept="image/jpeg, image/png, image/webp"
                        onChange={handleCustomUpload}
                    />
                </div>
            )}

            {/* Preview indicator */}
            {isGeneratingPreview && (
                <div className="flex items-center gap-2 text-xs text-purple-400 p-3 border-t border-border/50">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Generating preview...
                </div>
            )}
        </div>
    );
}
