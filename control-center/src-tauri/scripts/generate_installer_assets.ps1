$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function New-Bitmap($width, $height) {
    return New-Object System.Drawing.Bitmap($width, $height)
}

function Set-GraphicsQuality([System.Drawing.Graphics] $graphics) {
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
}

function Draw-RoundedRectangle(
    [System.Drawing.Graphics] $graphics,
    [System.Drawing.Pen] $pen,
    [System.Drawing.Brush] $brush,
    [float] $x,
    [float] $y,
    [float] $width,
    [float] $height,
    [float] $radius
) {
    $diameter = $radius * 2
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    try {
        $path.AddArc($x, $y, $diameter, $diameter, 180, 90)
        $path.AddArc($x + $width - $diameter, $y, $diameter, $diameter, 270, 90)
        $path.AddArc($x + $width - $diameter, $y + $height - $diameter, $diameter, $diameter, 0, 90)
        $path.AddArc($x, $y + $height - $diameter, $diameter, $diameter, 90, 90)
        $path.CloseFigure()

        if ($brush) {
            $graphics.FillPath($brush, $path)
        }
        if ($pen) {
            $graphics.DrawPath($pen, $path)
        }
    } finally {
        $path.Dispose()
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcTauriDir = Split-Path -Parent $scriptDir
$iconsDir = Join-Path $srcTauriDir "icons"
$windowsDir = Join-Path $srcTauriDir "windows"

$iconPath = Join-Path $iconsDir "icon.png"
$headerPath = Join-Path $windowsDir "installer-header.bmp"
$sidebarPath = Join-Path $windowsDir "installer-sidebar.bmp"

$brandBlack = [System.Drawing.Color]::FromArgb(255, 0, 0, 0)
$brandNavy = [System.Drawing.Color]::FromArgb(255, 15, 11, 115)
$brandBlue = [System.Drawing.Color]::FromArgb(255, 27, 81, 255)
$brandCyan = [System.Drawing.Color]::FromArgb(255, 96, 190, 255)
$panelGray = [System.Drawing.Color]::FromArgb(255, 208, 208, 214)
$panelWhite = [System.Drawing.Color]::FromArgb(255, 255, 255, 255)
$mutedBorder = [System.Drawing.Color]::FromArgb(255, 62, 62, 72)
$shadow = [System.Drawing.Color]::FromArgb(72, 0, 0, 0)

$icon = [System.Drawing.Image]::FromFile($iconPath)
try {
    $headerBitmap = New-Bitmap 150 57
    $headerGraphics = [System.Drawing.Graphics]::FromImage($headerBitmap)
    try {
        Set-GraphicsQuality $headerGraphics
        $headerGraphics.Clear($panelWhite)

        $accentBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
            (New-Object System.Drawing.Rectangle -ArgumentList 0, 0, 150, 57),
            $brandBlack,
            [System.Drawing.Color]::FromArgb(255, 12, 12, 16),
            [System.Drawing.Drawing2D.LinearGradientMode]::Horizontal
        )
        try {
            $headerGraphics.FillRectangle($accentBrush, 0, 0, 46, 57)
        } finally {
            $accentBrush.Dispose()
        }

        $headerGraphics.FillRectangle(
            (New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(20, 0, 0, 0))),
            0,
            52,
            150,
            5
        )

        $headerGraphics.DrawImage($icon, 6, 8, 32, 32)

        $titleFont = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
        $subtitleFont = New-Object System.Drawing.Font("Segoe UI", 7.8, [System.Drawing.FontStyle]::Regular)
        $titleBrush = New-Object System.Drawing.SolidBrush($brandNavy)
        $subtitleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 76, 84, 102))
        try {
            $headerGraphics.DrawString("gary4local", $titleFont, $titleBrush, 48, 8)
            $headerGraphics.DrawString("current-user install", $subtitleFont, $subtitleBrush, 50, 31)
        } finally {
            $titleFont.Dispose()
            $subtitleFont.Dispose()
            $titleBrush.Dispose()
            $subtitleBrush.Dispose()
        }
    } finally {
        $headerGraphics.Dispose()
    }
    $headerBitmap.Save($headerPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
    $headerBitmap.Dispose()

    $sidebarBitmap = New-Bitmap 164 314
    $sidebarGraphics = [System.Drawing.Graphics]::FromImage($sidebarBitmap)
    try {
        Set-GraphicsQuality $sidebarGraphics

        $sidebarBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
            (New-Object System.Drawing.Rectangle -ArgumentList 0, 0, 164, 314),
            $brandBlack,
            [System.Drawing.Color]::FromArgb(255, 16, 16, 20),
            [System.Drawing.Drawing2D.LinearGradientMode]::Vertical
        )
        try {
            $sidebarGraphics.FillRectangle($sidebarBrush, 0, 0, 164, 314)
        } finally {
            $sidebarBrush.Dispose()
        }

        $overlayBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(18, 96, 190, 255))
        try {
            $points1 = [System.Drawing.Point[]]@(
                (New-Object System.Drawing.Point -ArgumentList 0, 58),
                (New-Object System.Drawing.Point -ArgumentList 118, 0),
                (New-Object System.Drawing.Point -ArgumentList 164, 0),
                (New-Object System.Drawing.Point -ArgumentList 26, 84)
            )
            $points2 = [System.Drawing.Point[]]@(
                (New-Object System.Drawing.Point -ArgumentList 42, 314),
                (New-Object System.Drawing.Point -ArgumentList 164, 224),
                (New-Object System.Drawing.Point -ArgumentList 164, 314)
            )
            $sidebarGraphics.FillPolygon($overlayBrush, $points1)
            $sidebarGraphics.FillPolygon($overlayBrush, $points2)
        } finally {
            $overlayBrush.Dispose()
        }

        $shadowBrush = New-Object System.Drawing.SolidBrush($shadow)
        $frameBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 14, 14, 18))
        $framePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 72, 72, 82), 1.0)
        try {
            Draw-RoundedRectangle $sidebarGraphics $null $shadowBrush 27 35 110 110 20
            Draw-RoundedRectangle $sidebarGraphics $framePen $frameBrush 24 32 110 110 20
        } finally {
            $shadowBrush.Dispose()
            $frameBrush.Dispose()
            $framePen.Dispose()
        }

        $sidebarGraphics.DrawImage($icon, 35, 43, 88, 88)

        $bodyFont = New-Object System.Drawing.Font("Segoe UI", 9.2, [System.Drawing.FontStyle]::Bold)
        $titleBrush = New-Object System.Drawing.SolidBrush($panelWhite)
        $bodyBrush = New-Object System.Drawing.SolidBrush($panelGray)
        try {
            $sidebarGraphics.DrawString("localhost ctrl for gary4juce v3", $bodyFont, $bodyBrush, (New-Object System.Drawing.RectangleF -ArgumentList 18, 188, 128, 42))
        } finally {
            $bodyFont.Dispose()
            $titleBrush.Dispose()
            $bodyBrush.Dispose()
        }

        $borderPen = New-Object System.Drawing.Pen($mutedBorder, 1.0)
        try {
            $sidebarGraphics.DrawRectangle($borderPen, 0, 0, 163, 313)
        } finally {
            $borderPen.Dispose()
        }
    } finally {
        $sidebarGraphics.Dispose()
    }
    $sidebarBitmap.Save($sidebarPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
    $sidebarBitmap.Dispose()
} finally {
    $icon.Dispose()
}
