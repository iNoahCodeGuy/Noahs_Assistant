"""Supabase Storage service for file management.

This service handles:
- File uploads to Supabase Storage buckets
- Signed URL generation for secure file access
- Public and private bucket management

Buckets:
- public: Publicly accessible files (headshots, images)
- private: Private files requiring authentication (resumes, documents)

Usage:
    storage = StorageService()

    # Upload resume (private)
    url = storage.upload_resume('data/resume.pdf')

    # Upload headshot (public)
    url = storage.upload_headshot('data/headshot.jpg')

    # Generate signed URL (temporary access to private files)
    signed_url = storage.get_signed_url('resumes/noah_resume.pdf', expires_in=3600)
"""

import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from assistant.config.supabase_config import get_supabase_client, supabase_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Supabase Storage service for file management."""

    def __init__(self):
        """Initialize storage service with Supabase client."""
        self.client = get_supabase_client()
        self.public_bucket = supabase_settings.public_bucket
        self.private_bucket = supabase_settings.private_bucket

        logger.info(f"StorageService initialized with buckets: {self.public_bucket}, {self.private_bucket}")

    def upload_file(
        self,
        file_path: str,
        bucket: str,
        destination_path: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """Upload a file to Supabase Storage.

        Args:
            file_path: Path to local file to upload
            bucket: Target bucket name ('public' or 'private')
            destination_path: Path within bucket (defaults to filename)
            content_type: MIME type (auto-detected if not provided)

        Returns:
            Public URL for public bucket, path for private bucket

        Raises:
            FileNotFoundError: If local file doesn't exist
            Exception: If upload fails

        Example:
            url = storage.upload_file(
                'data/resume.pdf',
                bucket='private',
                destination_path='resumes/noah_resume.pdf'
            )
        """
        # Validate file exists
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Default destination path to filename
        if destination_path is None:
            destination_path = file_path_obj.name

        # Auto-detect content type
        if content_type is None:
            content_type = self._get_content_type(file_path_obj)

        try:
            # Read file contents
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # Upload to Supabase Storage
            response = self.client.storage.from_(bucket).upload(
                path=destination_path,
                file=file_data,
                file_options={"content-type": content_type}
            )

            logger.info(f"Uploaded {file_path} to {bucket}/{destination_path}")

            # Return appropriate URL
            if bucket == self.public_bucket:
                return self.get_public_url(bucket, destination_path)
            else:
                return destination_path

        except Exception as e:
            logger.error(f"Upload failed for {file_path}: {e}")
            raise Exception(f"Failed to upload file: {e}")

    def upload_resume(self, file_path: str, candidate_name: str = "noah") -> str:
        """Upload resume to private bucket.

        Args:
            file_path: Path to resume PDF file
            candidate_name: Name prefix for file (default: 'noah')

        Returns:
            Path to uploaded file (use get_signed_url for access)

        Example:
            path = storage.upload_resume('data/resume.pdf')
            url = storage.get_signed_url(path, expires_in=3600)
        """
        timestamp = datetime.now().strftime('%Y%m%d')
        destination = f"resumes/{candidate_name}_resume_{timestamp}.pdf"

        return self.upload_file(
            file_path=file_path,
            bucket=self.private_bucket,
            destination_path=destination,
            content_type='application/pdf'
        )

    def upload_headshot(self, file_path: str, candidate_name: str = "noah") -> str:
        """Upload headshot to public bucket.

        Args:
            file_path: Path to image file (jpg, png, webp)
            candidate_name: Name prefix for file (default: 'noah')

        Returns:
            Public URL to access the image

        Example:
            url = storage.upload_headshot('data/headshot.jpg')
            # URL can be used directly in <img src="...">
        """
        file_ext = Path(file_path).suffix
        destination = f"headshots/{candidate_name}_headshot{file_ext}"

        return self.upload_file(
            file_path=file_path,
            bucket=self.public_bucket,
            destination_path=destination
        )

    def get_public_url(self, bucket: str, file_path: str) -> str:
        """Get public URL for a file in public bucket.

        Args:
            bucket: Bucket name
            file_path: Path to file within bucket

        Returns:
            Public URL

        Note:
            Only works for files in public bucket. Use get_signed_url for private files.
        """
        return self.client.storage.from_(bucket).get_public_url(file_path)

    def get_signed_url(
        self,
        file_path: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600
    ) -> str:
        """Generate signed URL for temporary access to private files.

        Args:
            file_path: Path to file within bucket
            bucket: Bucket name (defaults to private bucket)
            expires_in: URL validity in seconds (default 1 hour)

        Returns:
            Signed URL valid for specified duration

        Use cases:
        - Email resume links that expire after download
        - Temporary access to private documents
        - Secure file sharing with hiring managers

        Example:
            # Generate 1-hour link
            url = storage.get_signed_url('resumes/noah_resume.pdf')

            # Generate 24-hour link
            url = storage.get_signed_url('resumes/noah_resume.pdf', expires_in=86400)
        """
        if bucket is None:
            bucket = self.private_bucket

        try:
            response = self.client.storage.from_(bucket).create_signed_url(
                path=file_path,
                expires_in=expires_in
            )

            signed_url = response.get('signedURL')
            if not signed_url:
                raise Exception("No signed URL returned from Supabase")

            logger.info(f"Generated signed URL for {file_path} (expires in {expires_in}s)")
            return signed_url

        except Exception as e:
            logger.error(f"Failed to generate signed URL for {file_path}: {e}")
            raise Exception(f"Failed to generate signed URL: {e}")

    def delete_file(self, file_path: str, bucket: Optional[str] = None) -> bool:
        """Delete a file from storage.

        Args:
            file_path: Path to file within bucket
            bucket: Bucket name (defaults to private bucket)

        Returns:
            True if deleted successfully

        Example:
            storage.delete_file('resumes/old_resume.pdf')
        """
        if bucket is None:
            bucket = self.private_bucket

        try:
            self.client.storage.from_(bucket).remove([file_path])
            logger.info(f"Deleted {file_path} from {bucket}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False

    def list_files(self, folder: str = "", bucket: Optional[str] = None) -> list:
        """List files in a bucket folder.

        Args:
            folder: Folder path within bucket (empty for root)
            bucket: Bucket name (defaults to private bucket)

        Returns:
            List of file metadata dicts

        Example:
            files = storage.list_files('resumes')
            for file in files:
                print(file['name'], file['created_at'])
        """
        if bucket is None:
            bucket = self.private_bucket

        try:
            response = self.client.storage.from_(bucket).list(folder)
            return response

        except Exception as e:
            logger.error(f"Failed to list files in {folder}: {e}")
            return []

    def _get_content_type(self, file_path: Path) -> str:
        """Auto-detect content type from file extension.

        Args:
            file_path: Path object of file

        Returns:
            MIME type string
        """
        extension_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.zip': 'application/zip',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }

        ext = file_path.suffix.lower()
        return extension_map.get(ext, 'application/octet-stream')

    def health_check(self) -> dict:
        """Check if storage service is healthy.

        Returns:
            Dict with status and bucket info

        Example:
            status = storage.health_check()
            if status['status'] == 'healthy':
                print("Storage service is operational")
        """
        try:
            # Try to list files in private bucket (minimal operation)
            self.client.storage.from_(self.private_bucket).list('')

            return {
                'status': 'healthy',
                'public_bucket': self.public_bucket,
                'private_bucket': self.private_bucket
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Global instance for convenience
_storage_service = None

def get_storage_service() -> StorageService:
    """Get or create global storage service instance.

    Returns:
        StorageService instance

    Example:
        from services import get_storage_service

        storage = get_storage_service()
        url = storage.upload_resume('data/resume.pdf')
    """
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
