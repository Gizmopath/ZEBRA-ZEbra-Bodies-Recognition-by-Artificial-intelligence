import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.roi.RectangleROI
import qupath.lib.gui.scripting.QPEx
import qupath.lib.common.GeneralTools

int numAnnotations = 5  // Specifica il numero di annotazioni da creare

// Ottieni immagine e server
def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()

// Definisci la dimensione della tile (in pixel)
int tileSizePixels = 512         // Tile size in pixels (fissa)
double requestedPixelSize = 0.5 // µm/pixel

// Calcola il downsampling
double pixelSize = server.getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

// Calcola la dimensione della tile in micrometri
int tileSizeFullRes = (int)(tileSizePixels * downsample)
int halfWidth = tileSizeFullRes / 2
int halfHeight = tileSizeFullRes / 2

// Ottieni il visualizzatore
def viewer = QPEx.getCurrentViewer()

// Numero di annotazioni da creare
double spacing = tileSizeFullRes * 1.5  // Distanza tra le annotazioni (in pixel)

// Crea le annotazioni
for (int i = 0; i < numAnnotations; i++) {
    // Calcola la posizione delle annotazioni (spostando lungo l'asse X)
    double cx = viewer.getCenterPixelX() + i * spacing
    double cy = viewer.getCenterPixelY()

    // Crea una nuova ROI rettangolare con la dimensione calcolata per la tile
    def roi = new RectangleROI(cx - halfWidth, cy - halfHeight, tileSizeFullRes, tileSizeFullRes)

    // Crea una nuova annotazione e aggiungila alla gerarchia degli oggetti
    def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass("Glomeruli"))
    imageData.getHierarchy().addObject(annotation, false)

    println "✔ Annotazione ${i + 1} creata"
}

println "✔ Tutte le annotazioni sono state create!"