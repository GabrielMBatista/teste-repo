# Performance Optimization Guide

## Quality Presets

### 🏃‍♂️ Speed Mode (2-4 minutes)

- **Resolution**: Up to 384x512
- **Frames**: Up to 16 frames
- **Duration**: Up to 2 seconds
- **Steps**: Up to 3 inference steps
- **FPS**: 8 fps
- **Audio Steps**: 10 steps
- **Precision**: float16
- **Best for**: Quick previews, testing prompts

### ⚖️ Balanced Mode (4-6 minutes)

- **Resolution**: Up to 480x640
- **Frames**: Up to 32 frames
- **Duration**: Up to 3 seconds
- **Steps**: Up to 4 inference steps
- **FPS**: 12 fps
- **Audio Steps**: 20 steps
- **Precision**: float16
- **Best for**: Good quality with reasonable speed

### 🚀 Quality Mode (6-10 minutes)

- **Resolution**: Up to 640x832
- **Frames**: Up to 48 frames
- **Duration**: Up to 4 seconds
- **Steps**: Up to 6 inference steps
- **FPS**: 16 fps
- **Audio Steps**: 30 steps
- **Precision**: bfloat16
- **Best for**: Best possible quality

## Manual Optimization Tips

### For Speed Priority:

1. Choose **Speed** preset
2. Set **float16** precision
3. Use **lower resolution** (384x512 or less)
4. **Fewer inference steps** (1-3)
5. **Shorter duration** (1-2 seconds)
6. **Lower FPS** (8 fps)
7. **Disable audio** or use fewer audio steps (5-10)

### For Quality Priority:

1. Choose **Quality** preset
2. Set **bfloat16** precision
3. Use **higher resolution** (up to 640x832)
4. **More inference steps** (4-6)
5. **Longer duration** (3-4 seconds)
6. **Higher FPS** (16-24 fps)
7. **Enable audio** with more steps (25-50)

### GPU Memory Management:

- **RTX 3060 (12GB)**: Can handle Quality mode at full settings
- **RTX 3070/3080**: Can exceed Quality mode limits
- **RTX 3050/1660**: Should stick to Speed or Balanced mode

## Real-time Adjustments

The interface allows you to:

- **Switch presets** and see limits update automatically
- **Fine-tune** individual parameters within preset limits
- **Monitor** estimated generation time
- **Balance** speed vs quality based on your needs

## Audio Optimization

### Fast Audio (5-15 steps):

- Basic ambient sounds
- Simple sound effects
- Quick generations

### Quality Audio (20-50 steps):

- Complex soundscapes
- Musical elements
- Synchronized sound effects

## Custom Combinations

You can mix and match settings:

- **High resolution + Low steps** = Detailed but simple motion
- **Low resolution + High steps** = Smooth motion with lower detail
- **Long duration + Low FPS** = Extended scenes with basic motion
- **Short duration + High FPS** = Smooth short clips
