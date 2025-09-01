from ImageSegmentationGUI import ImageSegmentationGUI

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import tkinterdnd2
    except ImportError:
        print("Installing required package: tkinterdnd2")
        import subprocess
        subprocess.check_call(["pip", "install", "tkinterdnd2"])
        import tkinterdnd2
    
    app = ImageSegmentationGUI()
    app.run()
