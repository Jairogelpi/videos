const fs = require('fs');
const path = require('path');

async function testUpload() {
    console.log("1. Testing Upload Flow (JS Mode)...");

    const dummyPath = path.join(__dirname, 'test_audio.mp3');
    fs.writeFileSync(dummyPath, 'fake audio content');
    const fileContent = fs.readFileSync(dummyPath);

    try {
        // 1. Sign
        console.log("2. Requesting Signed URL...");
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
            const txt = await signRes.text();
            throw new Error(`Sign Failed: ${signRes.status} ${txt}`);
        }

        const data = await signRes.json();
        console.log("3. Got Data:", data);
        const { uploadUrl } = data;

        // 2. Upload
        console.log("4. Uploading...");
        const uploadRes = await fetch(uploadUrl, {
            method: 'PUT',
            body: fileContent,
            headers: { 'Content-Type': 'audio/mpeg' }
        });

        if (!uploadRes.ok) {
            const txt = await uploadRes.text();
            throw new Error(`Upload Failed: ${uploadRes.status} ${txt}`);
        }

        console.log("5. SUCCESS");

    } catch (e) {
        console.error("ERROR:", e);
    } finally {
        if (fs.existsSync(dummyPath)) fs.unlinkSync(dummyPath);
    }
}

testUpload();
