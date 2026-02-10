-- Function to handle new user creation
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  -- 1. Create Public Profile
  insert into public.profiles (id, email, full_name)
  values (new.id, new.email, new.raw_user_meta_data ->> 'full_name');

  -- 2. Create Default Project for the user
  insert into public.projects (user_id, title, mood)
  values (new.id, 'My First Project', 'cinematic');

  return new;
end;
$$;

-- Trigger to call the function on new auth.users insert
-- Drop if exists to avoid errors on re-run
drop trigger if exists on_auth_user_created on auth.users;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
