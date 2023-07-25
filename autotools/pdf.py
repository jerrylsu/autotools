"""PDF tools.
"""
import os
import pdf2image
from loguru import logger
from typing import Optional, List
from tqdm import tqdm


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


def _convert_pdf_to_images(pdf_file_path: str) -> Optional[List]:
    """Convert PDF file to PIL Image objects.

    Args:
        pdf_file_path: The path of PDF file.
    """
    try:
        images = pdf2image.convert_from_path(pdf_file_path)
    except Exception as err:
        logger.info(f"An error occurred: {err}")
        return None
    return images


def convert_pdf_to_images(pdf_file_dir: str, output_dir: str) -> bool:
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
            images = _convert_pdf_to_images(pdf_file_path=file_path)
            if images:
                images_saved_path = os.path.join(output_dir, os.path.dirname(dirpath), filename)
                os.makedirs(images_saved_path, exist_ok=True)
                for idx, image in enumerate(images):
                    image.save(os.path.join(images_saved_path, str(idx)), "png")
    return True

