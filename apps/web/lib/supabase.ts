import { createClient } from '@supabase/supabase-js'

// Fallback to empty string for build; this likely won't be used on client anyway.
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || ''

export const supabase = createClient(supabaseUrl, supabaseServiceKey)
