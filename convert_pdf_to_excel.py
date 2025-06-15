import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import fitz  # PyMuPDF
import pandas as pd
import re
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
import arabic_reshaper
from bidi.algorithm import get_display

class PDFToExcelConverter:
    def __init__(self, pdf_path, output_dir="output"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.processed_images_dir = os.path.join(output_dir, "processed_images")
        self.ocr_text_dir = os.path.join(output_dir, "ocr_text")

        # Create directories
        for dir_path in [self.output_dir, self.images_dir, self.processed_images_dir, self.ocr_text_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Configure Tesseract (adjust path if needed)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def step1_pdf_to_images(self, max_pages=20):
        """Convert PDF pages to images"""
        print("Step 1: Converting PDF to images...")

        doc = fitz.open(self.pdf_path)
        total_pages = min(len(doc), max_pages)

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)  # High resolution
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Save image
            img_path = os.path.join(self.images_dir, f"page_{page_num+1:03d}.png")
            img.save(img_path, "PNG")
            print(f"Saved: page_{page_num+1:03d}.png")

        doc.close()
        print(f"Converted {total_pages} pages to images")

    def step2_rename_images(self):
        """Images are already properly named in step 1"""
        print("Step 2: Images already properly renamed")

    def detect_arabic_text_regions(self, image):
        """Detect Arabic text regions in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on size and aspect ratio
        arabic_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Arabic text typically has specific characteristics
            if w > 100 and h > 20 and w/h > 3:  # Horizontal text blocks
                arabic_regions.append((x, y, w, h))

        return arabic_regions

    def step3_remove_arabic_and_extras(self):
        """Remove Arabic text and extra information from images"""
        print("Step 3: Removing Arabic text and extra information...")

        for img_file in sorted(os.listdir(self.images_dir)):
            if img_file.endswith('.png'):
                img_path = os.path.join(self.images_dir, img_file)
                image = Image.open(img_path)

                # Convert to numpy array for OpenCV processing
                img_array = np.array(image)

                # Create mask for regions to remove
                mask = np.ones(img_array.shape[:2], dtype=np.uint8) * 255

                # Remove header title (top center area)
                height, width = img_array.shape[:2]
                # Remove top 15% of image (header area)
                mask[0:int(height*0.15), :] = 0

                # Remove bottom area with page numbers (bottom 10%)
                mask[int(height*0.9):, :] = 0

                # Detect and remove Arabic text regions
                arabic_regions = self.detect_arabic_text_regions(image)
                for x, y, w, h in arabic_regions:
                    # Check if this region likely contains Arabic text
                    region = img_array[y:y+h, x:x+w]
                    # Simple heuristic: if region has right-to-left text patterns
                    mask[y:y+h, x:x+w] = 0

                # Apply mask - set masked areas to white
                processed_img = img_array.copy()
                processed_img[mask == 0] = [255, 255, 255]  # White background

                # Save processed image
                processed_image = Image.fromarray(processed_img)
                output_path = os.path.join(self.processed_images_dir, img_file)
                processed_image.save(output_path)
                print(f"Processed: {img_file}")

        print("Arabic text and extras removed from all images")

    def step4_ocr_images(self):
        """Perform OCR on processed images"""
        print("Step 4: Performing OCR on images...")

        all_text = []

        for img_file in sorted(os.listdir(self.processed_images_dir)):
            if img_file.endswith('.png'):
                img_path = os.path.join(self.processed_images_dir, img_file)

                # Configure OCR for Bengali
                custom_config = r'--oem 3 --psm 6 -l ben+eng'

                try:
                    text = pytesseract.image_to_string(Image.open(img_path), config=custom_config)

                    # Save individual OCR result
                    text_file = os.path.join(self.ocr_text_dir, f"{img_file.replace('.png', '.txt')}")
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text)

                    all_text.append(f"=== PAGE {img_file} ===\n{text}\n")
                    print(f"OCR completed: {img_file}")

                except Exception as e:
                    print(f"OCR failed for {img_file}: {str(e)}")
                    all_text.append(f"=== PAGE {img_file} ===\n[OCR FAILED]\n")

        # Save combined OCR text
        combined_text_path = os.path.join(self.output_dir, "combined_ocr_text.txt")
        with open(combined_text_path, 'w', encoding='utf-8') as f:
            f.writelines(all_text)

        print("OCR completed for all images")
        return combined_text_path

    def step5_format_document(self, text_file_path):
        """Format the OCR text document"""
        print("Step 5: Formatting document...")

        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Clean up the text
        formatted_content = re.sub(r'\n+', '\n', content)  # Remove multiple newlines
        formatted_content = re.sub(r'\s+', ' ', formatted_content)  # Normalize spaces

        # Save formatted document
        formatted_doc_path = os.path.join(self.output_dir, "formatted_document.txt")
        with open(formatted_doc_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)

        print("Document formatting completed")
        return formatted_doc_path

    def extract_data_from_text(self, formatted_text_path):
        """Extract chapters, sections, and hadiths from formatted text"""
        print("Extracting data from formatted text...")

        with open(formatted_text_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chapters = []
        sections = []
        hadiths = []

        # Split content by pages
        pages = content.split('=== PAGE')

        chapter_id = 1
        section_id = 1
        hadith_id = 1

        for page in pages:
            if not page.strip():
                continue

            lines = page.split('\n')

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Extract chapters (অধ্যায়ঃ)
                if 'অধ্যায়ঃ' in line or 'অধ্যায়-' in line:
                    chapter_match = re.search(r'অধ্যায়[ঃ-]\s*(.+)', line)
                    if chapter_match:
                        chapter_name = chapter_match.group(1).strip()
                        chapters.append({
                            'id': chapter_id,
                            'name': chapter_name
                        })
                        chapter_id += 1

                # Extract sections (bold text after অধ্যায়ঃ)
                if line and len(line) > 20 and not line.startswith('[') and 'অধ্যায়' not in line:
                    # Check if this could be a section header
                    if re.match(r'^[^\[\]]+$', line) and not re.search(r'\d+\.\s', line):
                        sections.append({
                            'id': section_id,
                            'name': line.strip()
                        })
                        section_id += 1

                # Extract hadiths (text in [] brackets)
                hadith_matches = re.finditer(r'\[(\d+)\]', line)
                for match in hadith_matches:
                    hadith_number = match.group(1)
                    # Find the hadith text after the bracket
                    start_pos = match.end()

                    # Collect text until next hadith or end
                    hadith_text = ""
                    for j in range(i, min(i + 10, len(lines))):  # Look ahead up to 10 lines
                        text_line = lines[j].strip()
                        if j == i:
                            hadith_text += text_line[start_pos:].strip()
                        else:
                            if re.search(r'\[\d+\]', text_line):
                                break
                            hadith_text += " " + text_line

                    if hadith_text.strip():
                        hadiths.append({
                            'id': hadith_number,
                            'hadith': hadith_text.strip()
                        })

        print(f"Extracted: {len(chapters)} chapters, {len(sections)} sections, {len(hadiths)} hadiths")
        return chapters, sections, hadiths

    def step6_create_excel(self, chapters, sections, hadiths):
        """Create Excel file with three sheets"""
        print("Step 6: Creating Excel file...")

        # Create workbook
        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet

        # Create Chapter sheet
        chapter_sheet = wb.create_sheet(title="chapter")
        chapter_sheet['A1'] = 'id'
        chapter_sheet['B1'] = 'name'

        # Format headers
        for cell in ['A1', 'B1']:
            chapter_sheet[cell].font = Font(bold=True)
            chapter_sheet[cell].alignment = Alignment(horizontal='center')

        # Add chapter data
        for i, chapter in enumerate(chapters, 2):
            chapter_sheet[f'A{i}'] = chapter['id']
            chapter_sheet[f'B{i}'] = chapter['name']
            chapter_sheet[f'A{i}'].alignment = Alignment(horizontal='center')
            chapter_sheet[f'B{i}'].alignment = Alignment(horizontal='left')

        # Create Section sheet
        section_sheet = wb.create_sheet(title="section")
        section_sheet['A1'] = 'id'
        section_sheet['B1'] = 'name'

        # Format headers
        for cell in ['A1', 'B1']:
            section_sheet[cell].font = Font(bold=True)
            section_sheet[cell].alignment = Alignment(horizontal='center')

        # Add section data
        for i, section in enumerate(sections, 2):
            section_sheet[f'A{i}'] = section['id']
            section_sheet[f'B{i}'] = section['name']
            section_sheet[f'A{i}'].alignment = Alignment(horizontal='center')
            section_sheet[f'B{i}'].alignment = Alignment(horizontal='left')

        # Create Hadith sheet
        hadith_sheet = wb.create_sheet(title="hadith")
        hadith_sheet['A1'] = 'id'
        hadith_sheet['B1'] = 'hadith'

        # Format headers
        for cell in ['A1', 'B1']:
            hadith_sheet[cell].font = Font(bold=True)
            hadith_sheet[cell].alignment = Alignment(horizontal='center')

        # Add hadith data
        for i, hadith in enumerate(hadiths, 2):
            hadith_sheet[f'A{i}'] = hadith['id']
            hadith_sheet[f'B{i}'] = hadith['hadith']
            hadith_sheet[f'A{i}'].alignment = Alignment(horizontal='center')
            hadith_sheet[f'B{i}'].alignment = Alignment(horizontal='left')

        # Adjust column widths
        for sheet in [chapter_sheet, section_sheet, hadith_sheet]:
            sheet.column_dimensions['A'].width = 10
            sheet.column_dimensions['B'].width = 80

        # Save Excel file
        excel_path = os.path.join(self.output_dir, "hadith_data.xlsx")
        wb.save(excel_path)

        print(f"Excel file created: {excel_path}")
        return excel_path

    def run_conversion(self, max_pages=20):
        """Run the complete conversion process"""
        print("Starting PDF to Excel conversion...")
        print("="*50)

        try:
            # Step 1: PDF to Images
            self.step1_pdf_to_images(max_pages)

            # Step 2: Rename (already done in step 1)
            self.step2_rename_images()

            # Step 3: Remove Arabic text and extras
            self.step3_remove_arabic_and_extras()

            # Step 4: OCR
            ocr_text_path = self.step4_ocr_images()

            # Step 5: Format document
            formatted_doc_path = self.step5_format_document(ocr_text_path)

            # Extract data
            chapters, sections, hadiths = self.extract_data_from_text(formatted_doc_path)

            # Step 6: Create Excel
            excel_path = self.step6_create_excel(chapters, sections, hadiths)

            print("="*50)
            print("Conversion completed successfully!")
            print(f"Output directory: {self.output_dir}")
            print(f"Excel file: {excel_path}")
            print(f"OCR text: {ocr_text_path}")
            print(f"Formatted document: {formatted_doc_path}")

            return excel_path

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    # Initialize converter
    pdf_path = "bangla_hadith.pdf"  # Replace with your PDF path
    converter = PDFToExcelConverter(pdf_path)

    # Run conversion for first 20 pages
    excel_file = converter.run_conversion(max_pages=20)

    print(f"\nConversion completed! Excel file saved at: {excel_file}")
