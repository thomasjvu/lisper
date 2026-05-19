---
version: alpha
name: Lisper Gotham
description: A Gotham-inspired dark design system for a desktop-first speech-training app with a guided, game-like tone.
colors:
  primary: "#99d1ce"
  secondary: "#4e5166"
  tertiary: "#33859e"
  neutral: "#0c1014"
  surface: "#11151c"
  surface-alt: "#0f1b22"
  surface-soft: "#0a3749"
  border: "#195466"
  success: "#2aa889"
  warning: "#edb443"
  error: "#c23127"
typography:
  headline-lg:
    fontFamily: Segoe UI
    fontSize: 56px
    fontWeight: 800
    lineHeight: 1.04
    letterSpacing: -1px
  headline-md:
    fontFamily: Segoe UI
    fontSize: 32px
    fontWeight: 800
    lineHeight: 1.08
    letterSpacing: -0.8px
  body-md:
    fontFamily: Segoe UI
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
  body-lg:
    fontFamily: Segoe UI
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.5
  label-md:
    fontFamily: Segoe UI
    fontSize: 12px
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: 0.08em
rounded:
  sm: 10px
  md: 14px
  lg: 18px
  xl: 24px
  full: 999px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  xxl: 32px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "{colors.neutral}"
    rounded: "{rounded.md}"
    padding: 14px
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.primary}"
    rounded: "{rounded.md}"
    padding: 12px
  card-default:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.primary}"
    rounded: "{rounded.lg}"
    padding: 18px
  chip-default:
    backgroundColor: "{colors.surface-alt}"
    textColor: "{colors.primary}"
    rounded: "{rounded.full}"
    padding: 10px
---

# Lisper Gotham

## Overview

Lisper should feel like a clean desktop training game with clinical clarity, not a medical dashboard and not a kids app. The mood is focused, calm, slightly tactical, and low-anxiety. Guidance is explicit, but the tone remains concise and steady.

The interface should feel dense enough to be useful in one screen, with clear hierarchy and minimal decorative copy. The mascot can add warmth, but panels, navigation, and status surfaces should still feel professional.

## Colors

The palette is based on the Gotham terminal theme: deep blue-black backgrounds, cyan-teal text, muted slate metadata, and a restrained set of accent colors for progress and state.

- **Primary (#99d1ce):** Main readable text and high-importance foreground details.
- **Secondary (#4e5166):** Muted labels, metadata, and lower-priority copy.
- **Tertiary (#33859e):** Primary action fill and active UI emphasis.
- **Neutral (#0c1014):** Page background and deep negative space.
- **Surface (#11151c):** Main cards, shell panels, and persistent rails.
- **Surface Alt (#0f1b22):** Secondary containment for chips, metrics, and waveforms.
- **Surface Soft (#0a3749):** Selection and highlighted coaching areas.
- **Border (#195466):** Structural outlines and separators.
- **Success (#2aa889):** Positive progress and earned state.
- **Warning (#edb443):** Cautionary emphasis and intensity markers.
- **Error (#c23127):** Permission failures and hard blocked states.

## Typography

Typography stays local and system-based for performance. Use heavy, compressed-feeling hierarchy through weight and tracking instead of external fonts.

- **Headlines:** Bold and slightly tight. They should read like clear instructions, not marketing taglines.
- **Body:** Compact and readable. Prefer 14px or 16px with even line spacing.
- **Labels:** Uppercase or small-caps feeling, used sparingly for metadata and section markers.

## Layout

The app is desktop-first and should preserve a one-page command-center layout wherever possible. Main screens should fit without page scroll on common laptop sizes, using only inner-panel scrolling when space is constrained.

Spacing should stay compact and rhythmic. Use medium containment, not oversized whitespace. Navigation, content, and utility areas should align to a tight grid and avoid vertical drift when status text changes.

## Elevation & Depth

Depth should come from tonal layering, borders, and subtle contrast shifts rather than blurred glass or heavy shadows. Cards sit on darker backgrounds with crisp edge definition. Modal surfaces may use slightly stronger shadows, but standard app panels should remain flat and controlled.

## Shapes

The shape language is rounded but not bubbly. Use 10px to 24px radii for cards and controls. Full pills are reserved for chips, step trackers, and small status badges.

## Components

- **Primary buttons:** Gotham cyan fill with cyan-teal text. They are reserved for the main action in a region.
- **Secondary buttons:** Surface-colored with strong borders. They should never compete with the primary action.
- **Cards:** Dark surfaces with blue structural borders and compact internal spacing.
- **Progress bars:** Dark tracks with cyan fills. State color overrides are acceptable for warning and success.
- **Assessment and training capture panels:** Video/meter areas should use the darker alternate surface so live media stands apart from the surrounding shell.

## Do's and Don'ts

- Do keep instructions explicit and user-facing, especially in onboarding and assessment.
- Do use the darkest surfaces for layout structure and the cyan accent only for active emphasis.
- Do keep button labels short and direct.
- Don’t mix the old warm beige palette with the Gotham system.
- Don’t add decorative subtitles above every heading.
- Don’t make the mascot more visually dominant than the actual speech guidance.
