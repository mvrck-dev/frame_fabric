"use client";

import { useState, useEffect } from "react";
import { X, RotateCcw, Save, Loader2 } from "lucide-react";

interface PipelineConfig {
    dilation_px: number;
    feather_sigma: number;
    sdxl_steps: number;
    sdxl_guidance: number;
    sdxl_denoise: number;
    sdxl_scheduler: string;
    lora_scale: number;
    controlnet_scale: number;
    poisson_blend: boolean;
    color_transfer: boolean;
    esrgan_enabled: boolean;
    gan_enabled: boolean;
    [key: string]: number | string | boolean;
}

interface SettingsPanelProps {
    isOpen: boolean;
    onClose: () => void;
}

const DEFAULTS: PipelineConfig = {
    dilation_px: 5,
    feather_sigma: 2.0,
    sdxl_steps: 20,
    sdxl_guidance: 5.0,
    sdxl_denoise: 0.5,
    sdxl_scheduler: "euler",
    lora_scale: 0.7,
    controlnet_scale: 0.6,
    poisson_blend: true,
    color_transfer: true,
    esrgan_enabled: false,
    gan_enabled: false,
};

export default function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
    const [config, setConfig] = useState<PipelineConfig>(DEFAULTS);
    const [isSaving, setIsSaving] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);

    // Fetch current config on open
    useEffect(() => {
        if (!isOpen) return;

        const fetchConfig = async () => {
            try {
                const res = await fetch("http://localhost:8000/api/config");
                const data = await res.json();
                setConfig(prev => ({ ...prev, ...data }));
            } catch (err) {
                console.error("Failed to load config:", err);
            }
        };

        fetchConfig();
    }, [isOpen]);

    const handleSave = async () => {
        setIsSaving(true);
        setSaveSuccess(false);
        try {
            await fetch("http://localhost:8000/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(config),
            });
            setSaveSuccess(true);
            setTimeout(() => setSaveSuccess(false), 2000);
        } catch (err) {
            console.error("Failed to save config:", err);
        } finally {
            setIsSaving(false);
        }
    };

    const handleReset = () => {
        setConfig(DEFAULTS);
    };

    const updateField = (key: string, value: number | string | boolean) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    };

    if (!isOpen) return null;

    return (
        <>
            {/* Backdrop */}
            <div
                className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
                onClick={onClose}
            />

            {/* Panel */}
            <div className="fixed top-0 right-0 h-full w-96 bg-card/95 backdrop-blur-xl border-l border-border/50 z-50 flex flex-col animate-slide-in-right">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-border/50">
                    <h2 className="font-semibold text-sm tracking-wide">Pipeline Settings</h2>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-lg hover:bg-white/5 text-muted-foreground hover:text-foreground transition-colors"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>

                {/* Scrollable content */}
                <div className="flex-1 overflow-y-auto p-4 space-y-6">

                    {/* ── Mask Refinement ── */}
                    <section>
                        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                            Mask Refinement
                        </h3>
                        <div className="space-y-4">
                            <SliderField
                                label="Edge Dilation"
                                value={config.dilation_px}
                                min={1} max={12} step={1}
                                unit="px"
                                onChange={v => updateField("dilation_px", v)}
                            />
                            <SliderField
                                label="Feather Sigma"
                                value={config.feather_sigma}
                                min={0} max={8} step={0.5}
                                unit="σ"
                                onChange={v => updateField("feather_sigma", v)}
                            />
                        </div>
                    </section>

                    <hr className="border-border/30" />

                    {/* ── Export Quality ── */}
                    <section>
                        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                            Export Quality (SDXL)
                        </h3>
                        <div className="space-y-4">
                            <SliderField
                                label="Inference Steps"
                                value={config.sdxl_steps}
                                min={4} max={50} step={1}
                                onChange={v => updateField("sdxl_steps", v)}
                            />
                            <SliderField
                                label="Guidance Scale"
                                value={config.sdxl_guidance}
                                min={1} max={15} step={0.5}
                                onChange={v => updateField("sdxl_guidance", v)}
                            />
                            <SliderField
                                label="Denoise Strength"
                                value={config.sdxl_denoise}
                                min={0.1} max={1.0} step={0.05}
                                onChange={v => updateField("sdxl_denoise", v)}
                            />
                        </div>
                    </section>

                    <hr className="border-border/30" />

                    {/* ── Advanced ── */}
                    <section>
                        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                            Advanced
                        </h3>
                        <div className="space-y-4">
                            <div>
                                <label className="text-xs text-muted-foreground mb-1.5 block">Scheduler</label>
                                <div className="flex gap-2">
                                    {["euler", "lcm"].map(s => (
                                        <button
                                            key={s}
                                            onClick={() => updateField("sdxl_scheduler", s)}
                                            className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium uppercase tracking-wider transition-all ${
                                                config.sdxl_scheduler === s
                                                    ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                                                    : "bg-white/5 text-muted-foreground border border-white/8 hover:bg-white/8"
                                            }`}
                                        >
                                            {s}
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <SliderField
                                label="LoRA Scale"
                                value={config.lora_scale}
                                min={0} max={1.5} step={0.05}
                                onChange={v => updateField("lora_scale", v)}
                            />
                            <SliderField
                                label="ControlNet Scale"
                                value={config.controlnet_scale}
                                min={0} max={1.5} step={0.05}
                                onChange={v => updateField("controlnet_scale", v)}
                            />
                        </div>
                    </section>

                    <hr className="border-border/30" />

                    {/* ── Post-Processing ── */}
                    <section>
                        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                            Post-Processing
                        </h3>
                        <div className="space-y-3">
                            <ToggleField
                                label="Poisson Blending"
                                description="Seamless lighting continuity"
                                checked={config.poisson_blend}
                                onChange={v => updateField("poisson_blend", v)}
                            />
                            <ToggleField
                                label="Color Transfer"
                                description="Match scene illumination"
                                checked={config.color_transfer}
                                onChange={v => updateField("color_transfer", v)}
                            />
                            <ToggleField
                                label="ESRGAN Super-Resolution"
                                description="Upscale edited region (heavy)"
                                checked={config.esrgan_enabled}
                                onChange={v => updateField("esrgan_enabled", v)}
                            />
                            <ToggleField
                                label="GAN Live Preview"
                                description="Use SPADE-GAN (requires trained weights)"
                                checked={config.gan_enabled}
                                onChange={v => updateField("gan_enabled", v)}
                            />
                        </div>
                    </section>
                </div>

                {/* Footer actions */}
                <div className="p-4 border-t border-border/50 flex gap-2">
                    <button
                        onClick={handleReset}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-white/10 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all"
                    >
                        <RotateCcw className="w-3.5 h-3.5" />
                        Reset
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={isSaving}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                            saveSuccess
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                : "bg-purple-500/20 text-purple-300 border border-purple-500/30 hover:bg-purple-500/30"
                        }`}
                    >
                        {isSaving ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : saveSuccess ? (
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                        ) : (
                            <Save className="w-3.5 h-3.5" />
                        )}
                        {saveSuccess ? "Saved!" : "Save"}
                    </button>
                </div>
            </div>
        </>
    );
}

// ── Reusable Slider Field ──

function SliderField({
    label, value, min, max, step, unit, onChange,
}: {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    unit?: string;
    onChange: (v: number) => void;
}) {
    return (
        <div>
            <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs text-muted-foreground">{label}</label>
                <span className="text-xs font-mono text-foreground tabular-nums">
                    {Number.isInteger(step) ? value : value.toFixed(2)}
                    {unit && <span className="text-muted-foreground ml-0.5">{unit}</span>}
                </span>
            </div>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={e => onChange(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-white/10 accent-purple-500
                    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
                    [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:shadow-md
                    [&::-webkit-slider-thumb]:hover:bg-purple-400 [&::-webkit-slider-thumb]:transition-colors"
            />
        </div>
    );
}

// ── Reusable Toggle Field ──

function ToggleField({
    label, description, checked, onChange,
}: {
    label: string;
    description: string;
    checked: boolean;
    onChange: (v: boolean) => void;
}) {
    return (
        <div className="flex items-center justify-between py-1">
            <div>
                <p className="text-xs font-medium">{label}</p>
                <p className="text-[10px] text-muted-foreground">{description}</p>
            </div>
            <button
                onClick={() => onChange(!checked)}
                className={`relative w-9 h-5 rounded-full transition-colors ${
                    checked ? "bg-purple-500" : "bg-white/15"
                }`}
            >
                <span
                    className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${
                        checked ? "translate-x-4" : "translate-x-0"
                    }`}
                />
            </button>
        </div>
    );
}
