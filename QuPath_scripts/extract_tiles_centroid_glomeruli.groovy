selectAnnotations();
runPlugin('qupath.lib.plugins.objects.SplitAnnotationsPlugin', '{}')
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.images.servers.ImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.common.ColorTools
import qupath.lib.common.GeneralTools

// Get current image data & server
def imageData = getCurrentImageData()
def server = imageData.getServer()

// Define export settings
double requestedPixelSize = 0.5// µm/pixel
int tileSizePixels = 512         // Tile size in pixels (fixed)

// Output folder
def name = GeneralTools.stripExtension(server.getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'Export', name)
mkdirs(pathOutput)

// Get downsample factor
double pixelSize = server.getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

// Compute tile size in physical units (for consistent size in µm)
int tileSizeFullRes = (int)(tileSizePixels * downsample)
int halfWidth = tileSizeFullRes / 2
int halfHeight = tileSizeFullRes / 2

// Create labeled image server
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(1, ColorTools.BLACK)
    .downsample(downsample)
    .addLabel('Foamy', 0, ColorTools.WHITE)
    .addLabel('Glomerulus', 2, ColorTools.BLACK)
    .multichannelOutput(true)
    .build()

// Get relevant annotations
def tiles = getAnnotationObjects().findAll { it.getPathClass() == getPathClass("Glomeruli") }
println "Numero annotazioni trovate: " + tiles.size()

// Export tiles around each annotation
tiles.eachWithIndex { tile, index ->
    int centerX = tile.getROI().getCentroidX()
    int centerY = tile.getROI().getCentroidY()
    def tileClass = tile.getPathClass() != null ? tile.getPathClass().toString() : "Glomeruli"

    // Region request (original)
    def requestOrig = RegionRequest.createInstance(server.getPath(), downsample, centerX - halfWidth, centerY - halfHeight, tileSizeFullRes, tileSizeFullRes)
    def fileOrig = buildFilePath(pathOutput, "x${centerX}_y${centerY}_${tileClass}_${index + 1}.tif")
    writeImageRegion(server, requestOrig, fileOrig)

    // Region request (label)
    def requestLabel = RegionRequest.createInstance(labelServer.getPath(), downsample, centerX - halfWidth, centerY - halfHeight, tileSizeFullRes, tileSizeFullRes)
    def fileLabel = buildFilePath(pathOutput, "L_x${centerX}_y${centerY}_${tileClass}_${index + 1}.tif")
    writeImageRegion(labelServer, requestLabel, fileLabel)

    println "✔ Tile ${index + 1} salvata: ORIG + LABEL"
}

println "Done!"
