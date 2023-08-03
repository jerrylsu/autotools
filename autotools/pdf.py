"""PDF tools.
"""
import io
import os
from loguru import logger
from typing import List, Tuple, Optional, Union
from tqdm import tqdm
from PIL import Image
from fitz import Document, Page, Matrix

def download_pdf_from_url(url: str, save_pdf_file_path: str) -> bool:
    """Download PDF file from url.

    Args:
        url: The url of PDF file.
        save_pdf_file_path: The output file path for saving PDF.
    """
    status = False
    try:
        response = requests.get(url)
        status = response.status_code == 200
        if status:
            with open(save_pdf_file_path, "wb") as file:
                file.write(response.content)
            logger.info(f"PDF file saved to {save_pdf_file_path}")
        else:
            logger.info(f"Failed to download the PDF. Status code: {response.status_code}")
    except requests.exceptions.RequestException as err:
        logger.info(f"An error occurred: {err}")
        return status
    return status


def _page_to_image(page: Page, matrix: Matrix) -> Optional[Image.Image]:
    """Convert PDF Page object to PIL Image object.
    """
    try:
        page_image = page.get_pixmap(matrix=matrix)
        image_bytes = page_image.pil_tobytes("jpeg")
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
    except Exception as err:
        logger.info(f"Convert PDF Page to PIL Image object failed, error: {err}")
        return None
    return image


def pdf_to_images(
    file_path: Optional[str] = None,
    stream: Union[bytes, bytearray, io.BytesIO] = None,
    page_no: Optional[int] = None,
    scale_factor: int = 1,
    password: Optional[str] = None,
) -> Optional[List[Tuple[int, Image.Image]]]:
    """Convert PDF file to PIL Image object list.

    Args:
        page_no (`int`): The pdf page number to be converted, the page number starts from 0.
        scale_factor (`int`): The image scaling factor.
    """
    if not file_path and not stream:
        logger.error("Please pass in [file_path] or [stream] parameter.")
        return None

    try:
        document = Document(filename=file_path, stream=stream, filetype="pdf")
    except Exception as err:
        logger.info(f"fitz.Document failed to instantiate, error: {err}")
        return None

    if document.needs_pass:
        if document.authenticate(password) != 0:
            logger.error("The password of pdf is incorrect.")
            return None

    page_count = document.page_count
    page_no_list = list(range(page_count))
    if page_no:
        if page_no > page_count:
            logger.error("")
            return None
        page_no_list = [page_no]

    images = []
    for page_no in tqdm(page_no_list, total=len(page_no_list), desc="Convert:"):
        page = document.load_page(page_no)
        # The original image is shrunk when convertd from PDF by fitz,
        # so we scale the image size by scale_factor.
        matrix = Matrix(scale_factor, scale_factor)
        image = _page_to_image(page, matrix=matrix)
        images.append((page_no, image))
    return images


def convert_pdf_to_images(
    pdf_file_dir: str,
    output_dir: str,
    scale_factor: int = 2
) -> bool:
    """Convert PDF to PIL Image objects.

    Args:
        pdf_file_dir: The PDF directory.
        output_dir: The output directory for saving the images converted from PDF.
    """
    if not os.path.exists(pdf_file_dir):
        logger.warning(f"Not exist pdf_file_dir: {pdf_file_dir}.")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dirpath, dirnames, filenames in os.walk(pdf_file_dir):
        for filename in tqdm(filenames, total=len(filenames), desc="Convert:"):
            file_path = os.path.join(dirpath, filename)
            if not file_path.endswith((".pdf", ".PDF")):
                continue
            images = pdf_to_images(file_path=file_path, scale_factor=scale_factor)
            if images:
                images_saved_path = os.path.join(output_dir, os.path.dirname(dirpath), filename)
                os.makedirs(images_saved_path, exist_ok=True)
                for image in images:
                    image[1].save(os.path.join(images_saved_path, str(image[0])) + ".png", "png")
    return True

