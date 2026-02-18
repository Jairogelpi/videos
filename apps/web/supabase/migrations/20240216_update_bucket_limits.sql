-- Update the assets and outputs buckets to allow video/mp4, video/*, and application/octet-stream
-- This fixes the "mime type not supported" error in the workers.

update storage.buckets
set allowed_mime_types = array[
  'image/png', 
  'image/jpeg', 
  'image/gif', 
  'image/webp', 
  'audio/mpeg', 
  'audio/wav', 
  'video/mp4', 
  'video/x-m4v', 
  'video/quicktime',
  'video/webm',
  'application/octet-stream'
]
where id in ('assets', 'outputs');

-- Ensure the buckets are public (allows easier access for composition and preview)
update storage.buckets
set public = true
where id in ('assets', 'outputs');
