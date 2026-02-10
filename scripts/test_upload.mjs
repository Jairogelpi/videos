import fs from 'fs';
import path from 'path';

async function testUpload() {
    console.log("1. Testing Upload Flow...");

    // 1. Create a dummy file
    const dummyPath = path.join(__dirname, 'test_audio.mp3');
    fs.writeFileSync(dummyPath, 'fake audio content');
    const fileStats = fs.statSync(dummyPath);
    const fileContent = fs.readFileSync(dummyPath);

    try {
        // 2. Request Signed URL
        console.log("2. Requesting Signed URL from API...");
        const signRes = await fetch('http://localhost:3001/v1/uploads/sign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fileName: 'test_audio.mp3',
                contentType: 'audio/mpeg',
                userId: '00000000-0000-0000-0000-000000000000'
            })
        });

        if (!signRes.ok) {
            const err = await signRes.text();
            throw new Error(`API Sign Failed: ${signRes.status} ${err}`);
        }

        const { uploadUrl, assetPath } = await signRes.json();
        console.log(`3. Got Signed URL: ${uploadUrl.substring(0, 50)}...`);

        // 3. Perform Upload
        console.log("4. Uploading to Storage...");
        const uploadRes = await fetch(uploadUrl, {
            method: 'PUT',
            body: fileContent,
            headers: {
                'Content-Type': 'audio/mpeg',
                // Content-Length is auto-set by fetch usually, but explicit might help debug?
            }
        });

        if (!uploadRes.ok) {
            const errText = await uploadRes.text();
            throw new Error(`Storage Upload Failed: ${uploadRes.status} ${errText}`);
        }

        console.log("5. Upload SUCCESS!");
    } catch (e) {
        console.error("TEST FAILED:", e);
    } finally {
        if (fs.existsSync(dummyPath)) fs.unlinkSync(dummyPath);
    }
}

testUpload();
