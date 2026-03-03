"use client";

import { useState, useRef } from "react";
import { Upload, Image as ImageIcon, MapPin, Loader2 } from "lucide-react";

export default function EditorCanvas() {
    const [image, setImage] = useState<string | null>(null);
    const [isHovering, setIsHovering] = useState(false);
    const [pinCoords, setPinCoords] = useState<{ x: number, y: number } | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [maskImage, setMaskImage] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);

    const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            setImage(url);
            setPinCoords(null);
            setMaskImage(null);

            // Upload to backend to preload SAM memory
            setIsProcessing(true);
            try {
                const formData = new FormData();
                formData.append("file", file);

                const response = await fetch("http://localhost:8000/api/upload_image", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) throw new Error("Backend failed to encode image");
                console.log("Image successfully encoded into SAM memory.");
            } catch (error) {
                console.error("Error uploading image:", error);
                alert("Failed to connect to ML Backend. Make sure FastAPI is running!");
            } finally {
                setIsProcessing(false);
            }
        }
    };

    const handleCanvasClick = async (e: React.MouseEvent<HTMLDivElement>) => {
        if (!image || !imageRef.current) return;

        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        setPinCoords({ x, y });
        setIsProcessing(true);
        setMaskImage(null);

        try {
            // Send the click coords AND the current rendered width/height of the img tag
            // so the backend can scale the coordinates correctly to original image size
            const formData = new FormData();
            formData.append("x", x.toString());
            formData.append("y", y.toString());
            formData.append("width", imageRef.current.width.toString());
            formData.append("height", imageRef.current.height.toString());

            const response = await fetch("http://localhost:8000/api/segment", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Segmentation failed");

            const data = await response.json();
            setMaskImage(data.mask_base64);

        } catch (error) {
            console.error("Segmentation error:", error);
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="w-full h-full flex flex-col items-center justify-center relative">
            {!image ? (
                <div
                    className="w-full max-w-2xl aspect-video glass-card rounded-3xl border-2 border-dashed border-white/10 flex flex-col items-center justify-center cursor-pointer hover:border-purple-500/50 hover:bg-white/5 transition-all group"
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setIsHovering(true); }}
                    onDragLeave={() => setIsHovering(false)}
                    onDrop={async (e) => {
                        e.preventDefault();
                        setIsHovering(false);
                        const file = e.dataTransfer.files?.[0];
                        if (file) {
                            // Synthesize an event to reuse the logic
                            handleImageUpload({ target: { files: [file] } } as unknown as React.ChangeEvent<HTMLInputElement>);
                        }
                    }}
                    style={{ borderColor: isHovering ? "rgba(168, 85, 247, 0.5)" : "" }}
                >
                    <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 group-hover:bg-purple-500/20 transition-all">
                        <Upload className="w-8 h-8 text-muted-foreground group-hover:text-purple-400" />
                    </div>
                    <h3 className="text-xl font-semibold mb-2">Upload Room Photo</h3>
                    <p className="text-sm text-muted-foreground mb-6 text-center max-w-sm">
                        Drag and drop your image here, or click to browse. High-resolution images yield better generation results.
                    </p>
                    <div className="px-6 py-2 rounded-full bg-white/10 text-sm font-medium">
                        Browse Files
                    </div>
                </div>
            ) : (
                <div className="relative w-full h-full flex items-center justify-center group flex-col">
                    <div
                        className="relative shadow-2xl rounded-lg overflow-hidden border border-white/10 cursor-crosshair max-w-full max-h-full flex inline-block bg-black/50"
                        onClick={handleCanvasClick}
                    >
                        {/* Base Image */}
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                            ref={imageRef}
                            src={image}
                            alt="Room Upload"
                            className={`max-w-full max-h-[80vh] object-contain block pointer-events-none transition-all ${isProcessing && !pinCoords ? 'opacity-50 blur-sm' : ''}`}
                        />

                        {/* ML Mask Overlay */}
                        {maskImage && (
                            /* eslint-disable-next-line @next/next/no-img-element */
                            <img
                                src={maskImage}
                                alt="Segmentation Mask"
                                className="absolute inset-0 w-full h-full object-contain pointer-events-none animate-fade-in mix-blend-screen"
                            />
                        )}

                        {/* Loading / Encoding Spinner */}
                        {isProcessing && !pinCoords && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/20 pointer-events-none text-white">
                                <Loader2 className="w-12 h-12 animate-spin text-purple-400 mb-4" />
                                <span className="font-semibold drop-shadow-md">Encoding Semantic Features...</span>
                            </div>
                        )}

                        {/* Interactive Pin Marker */}
                        {pinCoords && (
                            <div
                                className="absolute w-6 h-6 -ml-3 -mt-6 pointer-events-none z-20 animate-bounce-short"
                                style={{ left: pinCoords.x, top: pinCoords.y }}
                            >
                                <MapPin className="text-purple-500 fill-purple-500 w-8 h-8 filter drop-shadow-lg" />
                                {isProcessing ? (
                                    <Loader2 className="absolute top-[8px] left-[13px] w-3 h-3 text-white animate-spin" />
                                ) : (
                                    <span className="absolute top-[8px] left-[13px] w-2 h-2 rounded-full bg-white animate-pulse"></span>
                                )}
                            </div>
                        )}

                        {/* Action Buttons Overlay */}
                        <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-30">
                            <button
                                className="bg-black/60 backdrop-blur-md p-2 rounded-lg border border-white/10 hover:bg-black/80 text-white transition-colors"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setImage(null);
                                    setPinCoords(null);
                                    setMaskImage(null);
                                }}
                                title="Replace Image"
                            >
                                <ImageIcon className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    {/* Floating Action Hint */}
                    {!pinCoords && !isProcessing && (
                        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 glass-card px-6 py-3 rounded-full border border-white/10 shadow-2xl pointer-events-none animate-fade-in-up">
                            <span className="text-sm font-medium tracking-wide flex items-center gap-3 text-white">
                                <span className="relative flex h-3 w-3">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500"></span>
                                </span>
                                Tap any surface to extract a segmentation mask
                            </span>
                        </div>
                    )}
                </div>
            )}

            <input
                type="file"
                className="hidden"
                ref={fileInputRef}
                accept="image/jpeg, image/png, image/webp"
                onChange={handleImageUpload}
            />
        </div>
    );
}
