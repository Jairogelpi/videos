-- Make bucket public
UPDATE storage.buckets SET public = true WHERE id = 'assets';

-- Drop existing restrictive policies
DROP POLICY IF EXISTS "Assets: Users manage own assets" ON storage.objects;

-- Create permissive policy for MVP (Allow all operations on assets bucket)
CREATE POLICY "Assets: Public Access" ON storage.objects
FOR ALL USING (bucket_id = 'assets') WITH CHECK (bucket_id = 'assets');
