-- =====================================================================
-- Caixa postal do Diagnóstico Acústico (Supabase)
--
-- Cole TUDO no "SQL Editor" do projeto no Supabase e clique em "Run".
-- Pode rodar de novo quantas vezes quiser: nada é duplicado.
--
-- Quem acessa o quê:
--   * cada técnico (conta criada pelo administrador) envia os próprios casos,
--     vê só os próprios casos e baixa os pacotes publicados;
--   * o site (computador da Central) usa a chave de serviço e acessa tudo.
-- =====================================================================

-- 1. Casos enviados pelos celulares ------------------------------------
create table if not exists public.casos (
    id                uuid primary key,
    tecnico_id        uuid not null default auth.uid() references auth.users (id) on delete cascade,
    tecnico_nome      text,
    criado_em         timestamptz not null default now(),
    enviado_em        timestamptz not null default now(),
    modelo_maquina    text not null,
    laudo             text not null default '',
    resultado_celular jsonb,
    arquivo           text not null,
    status            text not null default 'enviado'
                      check (status in ('enviado', 'recebido', 'aprovado', 'rejeitado')),
    -- resposta da Central ao técnico: primeiro só a análise ("status": "em_analise"),
    -- depois a decisão final ("aprovado" ou "rejeitado")
    resposta          jsonb,
    respondido_em     timestamptz
);
create index if not exists casos_status_idx on public.casos (status);
create index if not exists casos_tecnico_idx on public.casos (tecnico_id);

alter table public.casos enable row level security;

drop policy if exists "tecnico envia os proprios casos" on public.casos;
create policy "tecnico envia os proprios casos" on public.casos
    for insert to authenticated
    with check (tecnico_id = auth.uid() and status = 'enviado' and resposta is null);

drop policy if exists "tecnico ve os proprios casos" on public.casos;
create policy "tecnico ve os proprios casos" on public.casos
    for select to authenticated
    using (tecnico_id = auth.uid());

-- 2. Pacotes publicados para os celulares --------------------------------
create table if not exists public.publicacoes (
    versao    bigint primary key,
    criado_em timestamptz not null default now(),
    arquivo   text not null,
    sha256    text not null,
    tamanho   bigint not null,
    notas     text
);

alter table public.publicacoes enable row level security;

drop policy if exists "tecnicos leem as publicacoes" on public.publicacoes;
create policy "tecnicos leem as publicacoes" on public.publicacoes
    for select to authenticated
    using (true);

-- 3. Arquivos: pasta privada "caixa-postal" ------------------------------
--    casos/<id do técnico>/<id do caso>.zip   enviados pelo celular
--    publicacoes/pacote_vNNNN.zip             enviados pelo site
insert into storage.buckets (id, name, public)
values ('caixa-postal', 'caixa-postal', false)
on conflict (id) do nothing;

drop policy if exists "tecnico envia arquivos dos proprios casos" on storage.objects;
create policy "tecnico envia arquivos dos proprios casos" on storage.objects
    for insert to authenticated
    with check (
        bucket_id = 'caixa-postal'
        and (storage.foldername(name))[1] = 'casos'
        and (storage.foldername(name))[2] = auth.uid()::text
    );

drop policy if exists "tecnico le os proprios arquivos e as publicacoes" on storage.objects;
create policy "tecnico le os proprios arquivos e as publicacoes" on storage.objects
    for select to authenticated
    using (
        bucket_id = 'caixa-postal'
        and (
            (storage.foldername(name))[1] = 'publicacoes'
            or ((storage.foldername(name))[1] = 'casos' and (storage.foldername(name))[2] = auth.uid()::text)
        )
    );

-- 4. Base de casos compartilhada entre os computadores do site ---------
--    O que um computador alimenta, os outros recebem na sincronização.
--    Só o site (chave de serviço) acessa: os técnicos não veem estas tabelas
--    nem a pasta base/ de arquivos (as regras acima não dão acesso a elas).
create table if not exists public.base_casos (
    uuid          text primary key,
    dados         jsonb not null,
    laudo         text not null default '',
    arquivo       text,
    assinatura    text not null default '',
    computador    text,
    criado_em     timestamptz not null default now(),
    excluido      boolean not null default false,
    excluido_em   timestamptz
);
alter table public.base_casos enable row level security;

create table if not exists public.base_textos (
    chave         text primary key,
    categoria     text not null,
    textos        jsonb not null,
    atualizado_em text not null,
    computador    text
);
alter table public.base_textos enable row level security;

grant select, insert, update, delete on public.base_casos, public.base_textos to service_role;

-- avisa a API sobre as tabelas novas na hora
notify pgrst, 'reload schema';
