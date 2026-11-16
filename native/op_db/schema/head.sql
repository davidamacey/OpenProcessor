-- alembic revision: 0009_ingest_retry_at
--
-- PostgreSQL database dump
--

\restrict opschemahead

-- Dumped from database version 18.6 (Debian 18.6-1.pgdg13+2)
-- Dumped by pg_dump version 18.6 (Debian 18.6-1.pgdg13+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

-- *not* creating schema, since initdb creates it


--
-- Name: cluster_calibrations_immutable(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.cluster_calibrations_immutable() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
      BEGIN
        IF TG_OP = 'DELETE' AND OLD.project_id IS NOT NULL AND (
             NOT EXISTS (SELECT 1 FROM projects p WHERE p.id = OLD.project_id)
             OR op_project_delete_allows(OLD.project_id)) THEN
          RETURN OLD;
        END IF;
        RAISE EXCEPTION 'cluster_calibration_immutable: a calibration is never changed or deleted'
          USING ERRCODE = 'integrity_constraint_violation';
      END $$;


--
-- Name: cluster_children_project_from_parent(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.cluster_children_project_from_parent() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
      BEGIN
        SELECT project_id INTO NEW.project_id FROM cluster_partitions WHERE id = NEW.partition_id;
        RETURN NEW;
      END $$;


--
-- Name: cluster_children_ready_immutable(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.cluster_children_ready_immutable() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
      DECLARE
        parent_status text;
      BEGIN
        SELECT status INTO parent_status FROM cluster_partitions WHERE id = OLD.partition_id;
        IF parent_status IS NULL OR parent_status <> 'ready'
           OR (TG_OP = 'DELETE' AND op_project_delete_allows(OLD.project_id)) THEN
          RETURN CASE TG_OP WHEN 'DELETE' THEN OLD ELSE NEW END;
        END IF;
        RAISE EXCEPTION 'cluster_partition_immutable: % of a row of a ready partition', TG_OP
          USING ERRCODE = 'integrity_constraint_violation';
      END $$;


--
-- Name: cluster_partitions_ready_immutable(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.cluster_partitions_ready_immutable() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
      DECLARE
        kept text[] := ARRAY['status', 'job_id', 'parent_partition_id'];
      BEGIN
        IF OLD.status <> 'ready' THEN
          RETURN CASE TG_OP WHEN 'DELETE' THEN OLD ELSE NEW END;
        END IF;
        IF TG_OP = 'UPDATE' THEN
          IF (to_jsonb(NEW) - kept) = (to_jsonb(OLD) - kept)
             AND (NEW.job_id IS NULL OR NEW.job_id = OLD.job_id)
             AND (NEW.parent_partition_id IS NULL
                  OR NEW.parent_partition_id = OLD.parent_partition_id) THEN
            RETURN NEW;
          END IF;
          RAISE EXCEPTION 'cluster_partition_immutable: a ready partition only changes its status'
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        IF OLD.project_id IS NOT NULL AND (
             NOT EXISTS (SELECT 1 FROM projects p WHERE p.id = OLD.project_id)
             OR op_project_delete_allows(OLD.project_id)) THEN
          RETURN OLD;
        END IF;
        RAISE EXCEPTION 'cluster_partition_immutable: a ready partition is not deleted; discard it first'
          USING ERRCODE = 'integrity_constraint_violation';
      END $$;


--
-- Name: config_revisions_immutable(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.config_revisions_immutable() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
      BEGIN
        IF TG_OP = 'DELETE' THEN
          -- a cascade from the document, or the project_delete job removing a deleting project
          IF NOT EXISTS (SELECT 1 FROM config_documents d WHERE d.id = OLD.config_id)
             OR op_project_delete_allows(
                  (SELECT d.project_id FROM config_documents d WHERE d.id = OLD.config_id)) THEN
            RETURN OLD;
          END IF;
        END IF;
        RAISE EXCEPTION 'config_revision_immutable: revisions are never changed or deleted'
          USING ERRCODE = 'integrity_constraint_violation';
      END $$;


--
-- Name: embedding_model_versions_immutable(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.embedding_model_versions_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      DECLARE
        mutable text[] := ARRAY['status', 'engine_parity_min_cos'];
      BEGIN
        IF (to_jsonb(NEW) - mutable) IS DISTINCT FROM (to_jsonb(OLD) - mutable) THEN
          RAISE EXCEPTION 'embedding_model_version_immutable: only status and the parity change'
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        IF OLD.engine_parity_min_cos IS NOT NULL
           AND NEW.engine_parity_min_cos IS DISTINCT FROM OLD.engine_parity_min_cos THEN
          RAISE EXCEPTION 'embedding_model_version_immutable: the parity is recorded once; a lower parity is a new version'
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
      END $$;


--
-- Name: jobs_status_transition_guard(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.jobs_status_transition_guard() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
      BEGIN
        IF (OLD.status, NEW.status) NOT IN (('cancelling', 'cancelled'), ('cancelling', 'interrupted'), ('interrupted', 'cancelled'), ('interrupted', 'failed'), ('interrupted', 'queued'), ('paused', 'cancelled'), ('paused', 'queued'), ('queued', 'cancelled'), ('queued', 'failed'), ('queued', 'paused'), ('queued', 'running'), ('running', 'cancelling'), ('running', 'completed'), ('running', 'completed_with_errors'), ('running', 'failed'), ('running', 'interrupted'), ('running', 'paused'), ('running', 'queued')) THEN
          RAISE EXCEPTION 'invalid_job_transition: % -> %', OLD.status, NEW.status
            USING ERRCODE = 'check_violation';
        END IF;
        RETURN NEW;
      END $$;


--
-- Name: near_dup_children_project_from_run(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.near_dup_children_project_from_run() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
      BEGIN
        SELECT project_id INTO NEW.project_id FROM near_dup_runs WHERE id = NEW.run_id;
        RETURN NEW;
      END $$;


--
-- Name: op_model_project(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.op_model_project(mid uuid) RETURNS uuid
    LANGUAGE sql STABLE SECURITY DEFINER
    SET search_path TO 'public', 'pg_temp'
    AS $$
      SELECT project_id FROM models WHERE id = mid $$;


--
-- Name: op_project_delete_allows(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.op_project_delete_allows(pid uuid) RETURNS boolean
    LANGUAGE sql STABLE
    AS $$
      SELECT pid IS NOT NULL
        AND coalesce(current_setting('op.project_delete', true), '') = pid::text
        AND EXISTS (SELECT 1 FROM projects p WHERE p.id = pid AND p.status = 'deleting') $$;


--
-- Name: op_scope_allows(uuid); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.op_scope_allows(pid uuid) RETURNS boolean
    LANGUAGE sql STABLE
    AS $$
      SELECT op_scope_is_deployment() OR pid = op_scope_project()
             OR (pid IS NULL AND op_scope_is_core()) $$;


--
-- Name: op_scope_is_core(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.op_scope_is_core() RETURNS boolean
    LANGUAGE sql STABLE
    AS $$
      SELECT coalesce(current_setting('op.scope', true) = 'core', false) $$;


--
-- Name: op_scope_is_deployment(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.op_scope_is_deployment() RETURNS boolean
    LANGUAGE sql STABLE
    AS $$
      SELECT coalesce(current_setting('op.scope', true) = 'deployment', false) $$;


--
-- Name: op_scope_project(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.op_scope_project() RETURNS uuid
    LANGUAGE plpgsql STABLE
    AS $$
      BEGIN
        CASE current_setting('op.scope', true)
          WHEN 'project'    THEN RETURN nullif(current_setting('op.project_id', true), '')::uuid;
          WHEN 'deployment' THEN RETURN NULL;
          WHEN 'core'       THEN RETURN NULL;
          ELSE RAISE EXCEPTION 'scope_not_set' USING ERRCODE = 'insufficient_privilege';
        END CASE;
      END $$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


--
-- Name: assist_loop_budget_ledger; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_budget_ledger (
    id bigint NOT NULL,
    run_id uuid NOT NULL,
    request_id uuid NOT NULL,
    pool text NOT NULL,
    unit text NOT NULL,
    entry text NOT NULL,
    amount bigint NOT NULL,
    fencing_token bigint NOT NULL,
    at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_assist_loop_budget_ledger_entry CHECK ((entry = ANY (ARRAY['reserve'::text, 'settle'::text, 'release'::text, 'forfeit'::text])))
);

ALTER TABLE ONLY public.assist_loop_budget_ledger FORCE ROW LEVEL SECURITY;


--
-- Name: assist_loop_budget_ledger_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_budget_ledger ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.assist_loop_budget_ledger_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: assist_loop_budgets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_budgets (
    run_id uuid NOT NULL,
    pool text NOT NULL,
    unit text NOT NULL,
    limit_value bigint,
    reserved bigint DEFAULT '0'::bigint NOT NULL,
    spent bigint DEFAULT '0'::bigint NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_assist_loop_budgets_limit_value CHECK (((limit_value IS NULL) OR (limit_value >= 0))),
    CONSTRAINT ck_assist_loop_budgets_no_over_reserve CHECK (((unit = 'prompt_tokens'::text) OR (limit_value IS NULL) OR ((spent + reserved) <= limit_value))),
    CONSTRAINT ck_assist_loop_budgets_pool CHECK ((pool = ANY (ARRAY['vlm'::text, 'audit'::text]))),
    CONSTRAINT ck_assist_loop_budgets_reserved CHECK ((reserved >= 0)),
    CONSTRAINT ck_assist_loop_budgets_spent CHECK ((spent >= 0)),
    CONSTRAINT ck_assist_loop_budgets_unit CHECK ((unit = ANY (ARRAY['crops'::text, 'requests'::text, 'prompt_tokens'::text, 'completion_tokens'::text, 'wall_s'::text])))
);

ALTER TABLE ONLY public.assist_loop_budgets FORCE ROW LEVEL SECURITY;


--
-- Name: assist_loop_iterations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_iterations (
    run_id uuid NOT NULL,
    ordinal integer NOT NULL,
    status text NOT NULL,
    counts jsonb DEFAULT '{}'::jsonb NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    closed_at timestamp with time zone,
    CONSTRAINT ck_assist_loop_iterations_closed_iff_closed_at CHECK (((status = 'closed'::text) = (closed_at IS NOT NULL))),
    CONSTRAINT ck_assist_loop_iterations_ordinal CHECK ((ordinal >= 1)),
    CONSTRAINT ck_assist_loop_iterations_status CHECK ((status = ANY (ARRAY['open'::text, 'closed'::text])))
);

ALTER TABLE ONLY public.assist_loop_iterations FORCE ROW LEVEL SECURITY;


--
-- Name: assist_loop_nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_nodes (
    run_id uuid NOT NULL,
    path text NOT NULL,
    iteration integer NOT NULL,
    parent_path text,
    level smallint NOT NULL,
    member_count integer NOT NULL,
    eligible_count integer NOT NULL,
    membership_digest text NOT NULL,
    snapshot_storage_key text NOT NULL,
    step text NOT NULL,
    decision text,
    class_name text,
    answered integer,
    agree integer,
    bound double precision,
    distribution jsonb,
    admitted_crops integer,
    admitted_requests integer,
    label_op_id uuid,
    items_propagated integer DEFAULT 0 NOT NULL,
    skipped jsonb DEFAULT '{}'::jsonb NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_assist_loop_nodes_decision CHECK ((decision = ANY (ARRAY['pass'::text, 'split'::text, 'review_leaf'::text, 'review_small'::text, 'stale'::text, 'unvisited'::text]))),
    CONSTRAINT ck_assist_loop_nodes_done_iff_decided CHECK (((step = 'done'::text) = (decision IS NOT NULL))),
    CONSTRAINT ck_assist_loop_nodes_eligible_count CHECK ((eligible_count >= 0)),
    CONSTRAINT ck_assist_loop_nodes_level CHECK ((level >= 0)),
    CONSTRAINT ck_assist_loop_nodes_member_count CHECK ((member_count >= 0)),
    CONSTRAINT ck_assist_loop_nodes_path CHECK ((path ~ '^r[0-9]+(\.[0-9]+)*$'::text)),
    CONSTRAINT ck_assist_loop_nodes_step CHECK ((step = ANY (ARRAY['open'::text, 'admitted'::text, 'asked'::text, 'gated'::text, 'propagating'::text, 'done'::text])))
);

ALTER TABLE ONLY public.assist_loop_nodes FORCE ROW LEVEL SECURITY;


--
-- Name: assist_loop_proposals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_proposals (
    run_id uuid NOT NULL,
    item_id text NOT NULL,
    role text NOT NULL,
    proposer_kind text NOT NULL,
    proposer_ref text NOT NULL,
    request_id uuid NOT NULL,
    outcome text NOT NULL,
    class_name text,
    new_class_name text,
    attributes jsonb,
    item_write text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_assist_loop_proposals_answered_has_class CHECK (((outcome = 'answered'::text) = ((class_name IS NOT NULL) OR (new_class_name IS NOT NULL)))),
    CONSTRAINT ck_assist_loop_proposals_item_write CHECK ((item_write = ANY (ARRAY['written'::text, 'locked'::text, 'token_changed'::text, 'not_attempted'::text]))),
    CONSTRAINT ck_assist_loop_proposals_outcome CHECK ((outcome = ANY (ARRAY['answered'::text, 'unreadable'::text, 'failed'::text]))),
    CONSTRAINT ck_assist_loop_proposals_proposer_kind CHECK ((proposer_kind = 'vlm'::text)),
    CONSTRAINT ck_assist_loop_proposals_role CHECK ((role = ANY (ARRAY['primary'::text, 'audit'::text])))
);

ALTER TABLE ONLY public.assist_loop_proposals FORCE ROW LEVEL SECURITY;


--
-- Name: assist_loop_reviews; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_reviews (
    id uuid DEFAULT uuidv7() NOT NULL,
    run_id uuid NOT NULL,
    item_id text NOT NULL,
    queue text NOT NULL,
    machine_source text NOT NULL,
    machine_class_name text,
    shown_class_name text,
    decision text NOT NULL,
    human_class_name text,
    served_at timestamp with time zone NOT NULL,
    decided_at timestamp with time zone DEFAULT now() NOT NULL,
    active_ms integer,
    batch_size smallint DEFAULT '1'::smallint NOT NULL,
    label_op_id uuid,
    reviewer text NOT NULL,
    reviewer_user_id uuid,
    CONSTRAINT ck_assist_loop_reviews_active_ms CHECK (((active_ms >= 0) AND (active_ms <= 600000))),
    CONSTRAINT ck_assist_loop_reviews_batch_size CHECK (((batch_size >= 1) AND (batch_size <= 200))),
    CONSTRAINT ck_assist_loop_reviews_blind_audit_hides_class CHECK (((queue <> 'blind_audit'::text) OR (shown_class_name IS NULL))),
    CONSTRAINT ck_assist_loop_reviews_decision CHECK ((decision = ANY (ARRAY['accept'::text, 'correct'::text, 'reject'::text, 'skip'::text]))),
    CONSTRAINT ck_assist_loop_reviews_human_class_iff_accepted CHECK (((decision = ANY (ARRAY['accept'::text, 'correct'::text])) = (human_class_name IS NOT NULL))),
    CONSTRAINT ck_assist_loop_reviews_machine_source CHECK ((machine_source = ANY (ARRAY['propagated'::text, 'direct'::text, 'proposal_only'::text]))),
    CONSTRAINT ck_assist_loop_reviews_queue CHECK ((queue = ANY (ARRAY['verify'::text, 'blind_audit'::text])))
);

ALTER TABLE ONLY public.assist_loop_reviews FORCE ROW LEVEL SECURITY;


--
-- Name: assist_loop_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_runs (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    job_id uuid NOT NULL,
    preset_config_id uuid,
    preset_revision integer,
    request jsonb NOT NULL,
    strategy text NOT NULL,
    seed bigint NOT NULL,
    pinned jsonb NOT NULL,
    stop_reason text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    created_by_user_id uuid,
    CONSTRAINT ck_assist_loop_runs_preset_pair CHECK (((preset_config_id IS NULL) = (preset_revision IS NULL))),
    CONSTRAINT ck_assist_loop_runs_stop_reason CHECK ((stop_reason = ANY (ARRAY['done'::text, 'budget_crops'::text, 'budget_requests'::text, 'budget_tokens'::text, 'budget_time'::text, 'vlm_failed'::text]))),
    CONSTRAINT ck_assist_loop_runs_strategy CHECK ((strategy = 'cluster_sampled'::text))
);

ALTER TABLE ONLY public.assist_loop_runs FORCE ROW LEVEL SECURITY;


--
-- Name: assist_loop_selections; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assist_loop_selections (
    run_id uuid NOT NULL,
    item_id text NOT NULL,
    purpose text NOT NULL,
    node_path text,
    iteration integer NOT NULL,
    score double precision,
    draw_rank integer NOT NULL,
    selected_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_assist_loop_selections_purpose CHECK ((purpose = ANY (ARRAY['node_draw'::text, 'node_topup'::text, 'redraw'::text, 'audit'::text, 'blind_audit'::text])))
);

ALTER TABLE ONLY public.assist_loop_selections FORCE ROW LEVEL SECURITY;


--
-- Name: bakeoff_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bakeoff_runs (
    job_id uuid NOT NULL,
    project_id uuid NOT NULL,
    eval_dataset_key text NOT NULL,
    models jsonb NOT NULL,
    results_storage_key text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE ONLY public.bakeoff_runs FORCE ROW LEVEL SECURITY;


--
-- Name: class_registries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.class_registries (
    project_id uuid NOT NULL,
    next_class_id integer DEFAULT 0 NOT NULL,
    revision bigint DEFAULT '0'::bigint NOT NULL,
    migrated_from text,
    migrated_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_class_registries_migrated_from CHECK ((migrated_from = ANY (ARRAY['class_registry_file'::text, 'opensearch_registry_doc'::text]))),
    CONSTRAINT ck_class_registries_next_class_id CHECK ((next_class_id >= 0))
);

ALTER TABLE ONLY public.class_registries FORCE ROW LEVEL SECURITY;


--
-- Name: cluster_activation_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cluster_activation_events (
    id bigint NOT NULL,
    project_id uuid,
    target text NOT NULL,
    level text NOT NULL,
    embedding_field text NOT NULL,
    from_partition_id uuid,
    to_partition_id uuid NOT NULL,
    reason text NOT NULL,
    at timestamp with time zone DEFAULT now() NOT NULL,
    by text NOT NULL,
    CONSTRAINT ck_cluster_activation_events_reason CHECK ((reason = ANY (ARRAY['auto'::text, 'human'::text, 'rollback'::text, 'edit'::text, 'upgrade'::text])))
);

ALTER TABLE ONLY public.cluster_activation_events FORCE ROW LEVEL SECURITY;


--
-- Name: cluster_activation_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.cluster_activation_events ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.cluster_activation_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: cluster_activations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cluster_activations (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid,
    target text NOT NULL,
    level text NOT NULL,
    embedding_field text NOT NULL,
    serving_partition_id uuid NOT NULL,
    pending_partition_id uuid,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_by text NOT NULL,
    CONSTRAINT ck_cluster_activations_level CHECK ((level = ANY (ARRAY['class'::text, 'identity'::text, 'scene'::text]))),
    CONSTRAINT ck_cluster_activations_target CHECK ((target = ANY (ARRAY['item'::text, 'region_box'::text, 'face'::text, 'image'::text])))
);

ALTER TABLE ONLY public.cluster_activations FORCE ROW LEVEL SECURITY;


--
-- Name: cluster_backend_validations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cluster_backend_validations (
    id uuid DEFAULT uuidv7() NOT NULL,
    method text NOT NULL,
    backend_versions jsonb NOT NULL,
    embedding_field text NOT NULL,
    dataset_pin text NOT NULL,
    ari_vs_cpu real NOT NULL,
    purity_delta real NOT NULL,
    graph_recall real,
    passed boolean NOT NULL,
    validated_at timestamp with time zone DEFAULT now() NOT NULL,
    validated_by text NOT NULL
);


--
-- Name: cluster_calibrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cluster_calibrations (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid,
    target text NOT NULL,
    level text NOT NULL,
    embedding_field text NOT NULL,
    embedding_model text NOT NULL,
    preprocess_id text NOT NULL,
    kind text NOT NULL,
    fitted_on text NOT NULL,
    body jsonb NOT NULL,
    dev_metrics jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    CONSTRAINT ck_cluster_calibrations_kind CHECK ((kind = 'assign_bands'::text)),
    CONSTRAINT ck_cluster_calibrations_level CHECK ((level = ANY (ARRAY['class'::text, 'identity'::text, 'scene'::text]))),
    CONSTRAINT ck_cluster_calibrations_target CHECK ((target = ANY (ARRAY['item'::text, 'region_box'::text, 'face'::text, 'image'::text])))
);

ALTER TABLE ONLY public.cluster_calibrations FORCE ROW LEVEL SECURITY;


--
-- Name: cluster_identities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cluster_identities (
    cluster_uid uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid,
    target text NOT NULL,
    level text NOT NULL,
    name text,
    name_source text,
    anchored_at timestamp with time zone,
    anchored_by text,
    promoted_class_id integer,
    first_partition_id uuid,
    last_partition_id uuid,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    CONSTRAINT ck_cluster_identities_anchored_pair CHECK (((anchored_at IS NULL) = (anchored_by IS NULL))),
    CONSTRAINT ck_cluster_identities_level CHECK ((level = ANY (ARRAY['class'::text, 'identity'::text]))),
    CONSTRAINT ck_cluster_identities_name_source CHECK ((name_source = ANY (ARRAY['human'::text, 'import'::text, 'vlm'::text]))),
    CONSTRAINT ck_cluster_identities_target CHECK ((target = ANY (ARRAY['item'::text, 'region_box'::text, 'face'::text, 'image'::text])))
);

ALTER TABLE ONLY public.cluster_identities FORCE ROW LEVEL SECURITY;


--
-- Name: cluster_memberships; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cluster_memberships (
    partition_id uuid NOT NULL,
    project_id uuid,
    item_key text NOT NULL,
    cluster_uid uuid,
    state text NOT NULL,
    score real,
    margin real,
    CONSTRAINT ck_cluster_memberships_state CHECK ((state = ANY (ARRAY['assigned'::text, 'review'::text, 'unassigned'::text]))),
    CONSTRAINT ck_cluster_memberships_unassigned_iff_no_cluster CHECK (((state = 'unassigned'::text) = (cluster_uid IS NULL)))
);

ALTER TABLE ONLY public.cluster_memberships FORCE ROW LEVEL SECURITY;


--
-- Name: cluster_partitions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cluster_partitions (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid,
    target text NOT NULL,
    level text NOT NULL,
    embedding_field text NOT NULL,
    embedding_model text NOT NULL,
    preprocess_id text NOT NULL,
    method text NOT NULL,
    method_version text NOT NULL,
    profile_config_id uuid,
    profile_revision integer,
    params jsonb NOT NULL,
    calibration_id uuid,
    pool_filter jsonb NOT NULL,
    input_digest text NOT NULL,
    seed bigint NOT NULL,
    backend text NOT NULL,
    backend_detail jsonb NOT NULL,
    parent_partition_id uuid,
    job_id uuid,
    status text NOT NULL,
    n_items integer,
    n_clusters integer,
    n_review integer,
    n_unassigned integer,
    metrics jsonb DEFAULT '{}'::jsonb NOT NULL,
    diff jsonb,
    stage_durations jsonb DEFAULT '{}'::jsonb NOT NULL,
    peak_rss_mb integer,
    peak_vram_mb integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    created_by text NOT NULL,
    CONSTRAINT ck_cluster_partitions_backend CHECK ((backend = ANY (ARRAY['cpu'::text, 'gpu'::text]))),
    CONSTRAINT ck_cluster_partitions_level CHECK ((level = ANY (ARRAY['class'::text, 'identity'::text, 'scene'::text]))),
    CONSTRAINT ck_cluster_partitions_method CHECK ((method = ANY (ARRAY['ivf'::text, 'minibatch_kmeans'::text, 'union_find'::text, 'detector_class'::text, 'edit'::text]))),
    CONSTRAINT ck_cluster_partitions_profile_pair CHECK (((profile_config_id IS NULL) = (profile_revision IS NULL))),
    CONSTRAINT ck_cluster_partitions_status CHECK ((status = ANY (ARRAY['building'::text, 'ready'::text, 'failed'::text, 'discarded'::text]))),
    CONSTRAINT ck_cluster_partitions_target CHECK ((target = ANY (ARRAY['item'::text, 'region_box'::text, 'face'::text, 'image'::text])))
);

ALTER TABLE ONLY public.cluster_partitions FORCE ROW LEVEL SECURITY;


--
-- Name: combine_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.combine_runs (
    job_id uuid NOT NULL,
    target_project_id uuid NOT NULL,
    holdout_mode text NOT NULL,
    class_mapping jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_combine_runs_holdout_mode CHECK ((holdout_mode = ANY (ARRAY['union'::text, 'recompute'::text, 'drop'::text])))
);

ALTER TABLE ONLY public.combine_runs FORCE ROW LEVEL SECURITY;


--
-- Name: combine_sources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.combine_sources (
    job_id uuid NOT NULL,
    "position" smallint NOT NULL,
    source_project_id uuid,
    source_slug text NOT NULL,
    CONSTRAINT ck_combine_sources_position CHECK ((("position" >= 0) AND ("position" <= 7)))
);

ALTER TABLE ONLY public.combine_sources FORCE ROW LEVEL SECURITY;


--
-- Name: config_activation_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.config_activation_events (
    id bigint NOT NULL,
    project_id uuid NOT NULL,
    axis text NOT NULL,
    from_config_id uuid,
    from_revision integer,
    to_config_id uuid,
    to_revision integer,
    reason text NOT NULL,
    at timestamp with time zone DEFAULT now() NOT NULL,
    by text NOT NULL,
    CONSTRAINT ck_config_activation_events_reason CHECK ((reason = ANY (ARRAY['activate'::text, 'rollback'::text, 'clone'::text, 'upgrade'::text]))),
    CONSTRAINT ck_config_activation_events_to_pair CHECK (((to_config_id IS NULL) = (to_revision IS NULL)))
);

ALTER TABLE ONLY public.config_activation_events FORCE ROW LEVEL SECURITY;


--
-- Name: config_activation_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.config_activation_events ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.config_activation_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: config_activations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.config_activations (
    project_id uuid NOT NULL,
    axis text NOT NULL,
    kind text NOT NULL,
    config_id uuid NOT NULL,
    revision integer NOT NULL,
    activated_at timestamp with time zone DEFAULT now() NOT NULL,
    activated_by text NOT NULL,
    CONSTRAINT ck_config_activations_axis CHECK ((axis = ANY (ARRAY['prompt_pack'::text, 'detection_profile'::text, 'vlm'::text, 'open_vocab'::text, 'ingest_profile'::text, 'clustering_item'::text, 'clustering_region_box'::text, 'assist_loop_preset'::text, 'near_dup'::text]))),
    CONSTRAINT ck_config_activations_axis_kind CHECK (((((((((((axis = 'prompt_pack'::text) AND (kind = 'prompt_pack'::text)) OR ((axis = 'detection_profile'::text) AND (kind = 'region_profile'::text))) OR ((axis = 'vlm'::text) AND (kind = 'vlm_endpoint'::text))) OR ((axis = 'open_vocab'::text) AND (kind = 'open_vocab_set'::text))) OR ((axis = 'ingest_profile'::text) AND (kind = 'ingest_profile'::text))) OR ((axis = 'clustering_item'::text) AND (kind = 'cluster_profile'::text))) OR ((axis = 'clustering_region_box'::text) AND (kind = 'cluster_profile'::text))) OR ((axis = 'assist_loop_preset'::text) AND (kind = 'assist_loop_preset'::text))) OR ((axis = 'near_dup'::text) AND (kind = 'near_dup_profile'::text))))
);

ALTER TABLE ONLY public.config_activations FORCE ROW LEVEL SECURITY;


--
-- Name: config_documents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.config_documents (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid,
    kind text NOT NULL,
    name text NOT NULL,
    current_revision integer NOT NULL,
    description text DEFAULT ''::text NOT NULL,
    cloned_from_id uuid,
    cloned_from_revision integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_by text NOT NULL,
    updated_by_user_id uuid,
    CONSTRAINT ck_config_documents_current_revision CHECK ((current_revision >= 1)),
    CONSTRAINT ck_config_documents_kind CHECK ((kind = ANY (ARRAY['prompt_pack'::text, 'region_profile'::text, 'open_vocab_set'::text, 'ingest_profile'::text, 'vlm_endpoint'::text, 'serving_profile'::text, 'cluster_profile'::text, 'assist_loop_preset'::text, 'near_dup_profile'::text]))),
    CONSTRAINT ck_config_documents_name CHECK ((name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'::text)),
    CONSTRAINT ck_config_documents_scope CHECK (((kind = ANY (ARRAY['vlm_endpoint'::text, 'serving_profile'::text])) = (project_id IS NULL)))
);

ALTER TABLE ONLY public.config_documents FORCE ROW LEVEL SECURITY;


--
-- Name: config_revisions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.config_revisions (
    config_id uuid NOT NULL,
    revision integer NOT NULL,
    body jsonb NOT NULL,
    body_sha256 text NOT NULL,
    description text DEFAULT ''::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    CONSTRAINT ck_config_revisions_revision CHECK ((revision >= 1))
);

ALTER TABLE ONLY public.config_revisions FORCE ROW LEVEL SECURITY;


--
-- Name: config_runtime_reports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.config_runtime_reports (
    project_id uuid NOT NULL,
    process text NOT NULL,
    host text NOT NULL,
    config_epoch bigint NOT NULL,
    applied_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE ONLY public.config_runtime_reports FORCE ROW LEVEL SECURITY;


--
-- Name: dataset_exports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset_exports (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    job_id uuid,
    kind text NOT NULL,
    class_names text[] NOT NULL,
    manifest_sha256 text NOT NULL,
    artifact_prefix text NOT NULL,
    split_seed bigint NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_dataset_exports_kind CHECK ((kind = ANY (ARRAY['yolo'::text, 'single_class'::text])))
);

ALTER TABLE ONLY public.dataset_exports FORCE ROW LEVEL SECURITY;


--
-- Name: dataset_import_images; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset_import_images (
    import_id uuid NOT NULL,
    image_key text NOT NULL,
    image_id text,
    item_ids text[] DEFAULT '{}'::text[] NOT NULL,
    status text NOT NULL,
    error text,
    detail jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT ck_dataset_import_images_status CHECK ((status = ANY (ARRAY['pending'::text, 'written'::text, 'skipped'::text, 'failed'::text, 'undone'::text])))
);

ALTER TABLE ONLY public.dataset_import_images FORCE ROW LEVEL SECURITY;


--
-- Name: dataset_imports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset_imports (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    job_id uuid,
    undo_job_id uuid,
    upload_id uuid,
    source_key text,
    format text NOT NULL,
    mapping jsonb NOT NULL,
    options jsonb NOT NULL,
    scan jsonb,
    report jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    undone_at timestamp with time zone,
    CONSTRAINT ck_dataset_imports_one_source CHECK (((upload_id IS NULL) <> (source_key IS NULL)))
);

ALTER TABLE ONLY public.dataset_imports FORCE ROW LEVEL SECURITY;


--
-- Name: dataset_uploads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset_uploads (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    filename text NOT NULL,
    size_bytes bigint NOT NULL,
    sha256 text,
    storage_key text NOT NULL,
    status text NOT NULL,
    extract_job_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    CONSTRAINT ck_dataset_uploads_size_bytes CHECK ((size_bytes >= 0)),
    CONSTRAINT ck_dataset_uploads_status CHECK ((status = ANY (ARRAY['receiving'::text, 'received'::text, 'extracting'::text, 'extracted'::text, 'failed'::text, 'expired'::text])))
);

ALTER TABLE ONLY public.dataset_uploads FORCE ROW LEVEL SECURITY;


--
-- Name: deployment_counters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.deployment_counters (
    name text NOT NULL,
    value bigint DEFAULT '0'::bigint NOT NULL
);


--
-- Name: deployment_settings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.deployment_settings (
    key text NOT NULL,
    value jsonb NOT NULL,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_by text NOT NULL,
    CONSTRAINT ck_deployment_settings_key CHECK ((key = ANY (ARRAY['local_vlm_desired'::text, 'vlm_paused_for_training'::text, 'active_serving_profile'::text, 'upgrade_state'::text])))
);


--
-- Name: embedding_evaluation_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.embedding_evaluation_runs (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    job_id uuid,
    baseline_space_id uuid NOT NULL,
    candidate_space_id uuid NOT NULL,
    protocol text NOT NULL,
    sample_spec jsonb NOT NULL,
    metrics jsonb NOT NULL,
    verdict text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    CONSTRAINT ck_embedding_evaluation_runs_distinct_spaces CHECK ((baseline_space_id <> candidate_space_id)),
    CONSTRAINT ck_embedding_evaluation_runs_protocol CHECK ((protocol = ANY (ARRAY['label_consistency_v1'::text, 'query_set_v1'::text]))),
    CONSTRAINT ck_embedding_evaluation_runs_verdict CHECK ((verdict = ANY (ARRAY['better'::text, 'tie'::text, 'worse'::text, 'inconclusive'::text])))
);

ALTER TABLE ONLY public.embedding_evaluation_runs FORCE ROW LEVEL SECURITY;


--
-- Name: embedding_model_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.embedding_model_versions (
    id uuid DEFAULT uuidv7() NOT NULL,
    model_key text NOT NULL,
    version integer NOT NULL,
    modalities text[] NOT NULL,
    native_dim integer NOT NULL,
    matryoshka_dims integer[] DEFAULT '{}'::integer[] NOT NULL,
    distance text NOT NULL,
    input_spec jsonb NOT NULL,
    input_spec_sha256 text NOT NULL,
    weights_ref text NOT NULL,
    weights_sha256 text NOT NULL,
    "precision" text NOT NULL,
    triton_image_model text,
    triton_text_model text,
    engine_parity_min_cos real,
    licence text NOT NULL,
    status text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    CONSTRAINT ck_embedding_model_versions_distance CHECK ((distance = 'cosine'::text)),
    CONSTRAINT ck_embedding_model_versions_image_model CHECK (((triton_image_model IS NOT NULL) = ('image'::text = ANY (modalities)))),
    CONSTRAINT ck_embedding_model_versions_modalities CHECK (((modalities <@ ARRAY['image'::text, 'text'::text]) AND (cardinality(modalities) >= 1))),
    CONSTRAINT ck_embedding_model_versions_model_key CHECK ((model_key ~ '^[a-z][a-z0-9_]{1,39}$'::text)),
    CONSTRAINT ck_embedding_model_versions_native_dim CHECK (((native_dim >= 8) AND (native_dim <= 4096))),
    CONSTRAINT ck_embedding_model_versions_parity CHECK (((engine_parity_min_cos >= (0)::double precision) AND (engine_parity_min_cos <= (1)::double precision))),
    CONSTRAINT ck_embedding_model_versions_precision CHECK (("precision" = ANY (ARRAY['fp32'::text, 'bf16'::text, 'fp16'::text, 'int8'::text]))),
    CONSTRAINT ck_embedding_model_versions_status CHECK ((status = ANY (ARRAY['available'::text, 'deprecated'::text, 'retired'::text]))),
    CONSTRAINT ck_embedding_model_versions_text_model CHECK (((triton_text_model IS NOT NULL) = ('text'::text = ANY (modalities)))),
    CONSTRAINT ck_embedding_model_versions_version CHECK ((version >= 1))
);


--
-- Name: embedding_reindex_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.embedding_reindex_runs (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    space_id uuid NOT NULL,
    job_id uuid NOT NULL,
    mode text NOT NULL,
    scope jsonb DEFAULT '{}'::jsonb NOT NULL,
    docs_total bigint,
    docs_done bigint DEFAULT '0'::bigint NOT NULL,
    docs_failed bigint DEFAULT '0'::bigint NOT NULL,
    docs_skipped bigint DEFAULT '0'::bigint NOT NULL,
    bytes_estimate bigint,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    CONSTRAINT ck_embedding_reindex_runs_mode CHECK ((mode = ANY (ARRAY['build'::text, 'backfill'::text, 'clear'::text])))
);

ALTER TABLE ONLY public.embedding_reindex_runs FORCE ROW LEVEL SECURITY;


--
-- Name: embedding_spaces; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.embedding_spaces (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    space_key text NOT NULL,
    model_version_id uuid NOT NULL,
    target text NOT NULL,
    dim integer NOT NULL,
    role text NOT NULL,
    status text NOT NULL,
    index_role text NOT NULL,
    vector_field text NOT NULL,
    state_field text,
    nested_path text,
    knn_method jsonb NOT NULL,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    built_by_job_id uuid,
    ready_at timestamp with time zone,
    retired_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    CONSTRAINT ck_embedding_spaces_box_has_nested_path CHECK (((target = 'box'::text) = (nested_path IS NOT NULL))),
    CONSTRAINT ck_embedding_spaces_dim CHECK (((dim >= 8) AND (dim <= 4096))),
    CONSTRAINT ck_embedding_spaces_frame_in_images CHECK (((target = 'frame'::text) = (index_role = 'images'::text))),
    CONSTRAINT ck_embedding_spaces_index_role CHECK ((index_role = ANY (ARRAY['images'::text, 'items'::text]))),
    CONSTRAINT ck_embedding_spaces_primary_is_ready CHECK (((role <> 'primary'::text) OR (status = 'ready'::text))),
    CONSTRAINT ck_embedding_spaces_role CHECK ((role = ANY (ARRAY['primary'::text, 'optional'::text, 'candidate'::text]))),
    CONSTRAINT ck_embedding_spaces_space_key CHECK ((space_key ~ '^[a-z][a-z0-9_]{2,47}$'::text)),
    CONSTRAINT ck_embedding_spaces_status CHECK ((status = ANY (ARRAY['pending'::text, 'building'::text, 'ready'::text, 'retiring'::text, 'retired'::text, 'failed'::text]))),
    CONSTRAINT ck_embedding_spaces_target CHECK ((target = ANY (ARRAY['item'::text, 'box'::text, 'frame'::text, 'item_text'::text])))
);

ALTER TABLE ONLY public.embedding_spaces FORCE ROW LEVEL SECURITY;


--
-- Name: events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.events (
    id bigint NOT NULL,
    xid xid8 DEFAULT pg_current_xact_id() NOT NULL,
    project_id uuid,
    job_id uuid,
    type text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_events_payload_size CHECK ((octet_length((payload)::text) <= 7000))
)
WITH (autovacuum_vacuum_scale_factor='0.05', autovacuum_vacuum_insert_scale_factor='0.05');

ALTER TABLE ONLY public.events FORCE ROW LEVEL SECURITY;


--
-- Name: events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.events ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: gpu_devices; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gpu_devices (
    gpu_uuid text NOT NULL,
    host text NOT NULL,
    nvml_index integer NOT NULL,
    cuda_index integer,
    name text NOT NULL,
    compute_capability text NOT NULL,
    memory_total_mib integer NOT NULL,
    numa_node integer,
    pcie_gen integer,
    pcie_width integer,
    hw_jpeg_decoder boolean NOT NULL,
    driver_version text NOT NULL,
    cuda_version text NOT NULL,
    seen_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: gpu_reservations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gpu_reservations (
    gpu_uuid text NOT NULL,
    holder_job_id uuid NOT NULL,
    reserved_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE ONLY public.gpu_reservations FORCE ROW LEVEL SECURITY;


--
-- Name: holdout_freezes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.holdout_freezes (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    is_current boolean NOT NULL,
    source text NOT NULL,
    item_count integer NOT NULL,
    seed bigint,
    snapshot_storage_key text,
    frozen_at timestamp with time zone DEFAULT now() NOT NULL,
    frozen_by text NOT NULL,
    near_dup_image_run_id uuid,
    near_dup_item_run_id uuid,
    moved_by_near_dup integer DEFAULT 0 NOT NULL
);

ALTER TABLE ONLY public.holdout_freezes FORCE ROW LEVEL SECURITY;


--
-- Name: holdout_near_dup_moves; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.holdout_near_dup_moves (
    id bigint NOT NULL,
    item_id text NOT NULL,
    anchor_item_id text NOT NULL,
    group_id uuid,
    freeze_id uuid,
    run_id uuid,
    moved_at timestamp with time zone DEFAULT now() NOT NULL,
    project_id uuid NOT NULL
);

ALTER TABLE ONLY public.holdout_near_dup_moves FORCE ROW LEVEL SECURITY;


--
-- Name: holdout_near_dup_moves_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.holdout_near_dup_moves ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.holdout_near_dup_moves_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: image_contents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.image_contents (
    id bigint NOT NULL,
    imohash text NOT NULL,
    content_sha256 text,
    image_id text NOT NULL,
    image_path text NOT NULL,
    size_bytes bigint NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    project_id uuid,
    CONSTRAINT ck_image_contents_imohash CHECK ((imohash ~ '^[0-9a-f]{32}$'::text)),
    CONSTRAINT ck_image_contents_sha256 CHECK (((content_sha256 IS NULL) OR (content_sha256 ~ '^[0-9a-f]{64}$'::text))),
    CONSTRAINT ck_image_contents_size CHECK ((size_bytes >= 0))
);

ALTER TABLE ONLY public.image_contents FORCE ROW LEVEL SECURITY;


--
-- Name: image_contents_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.image_contents ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.image_contents_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: ingest_source_files; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingest_source_files (
    source_id uuid NOT NULL,
    file_key text NOT NULL,
    status text NOT NULL,
    job_id uuid,
    attempts integer DEFAULT 0 NOT NULL,
    image_id text,
    error text,
    enqueued_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    retry_at timestamp with time zone,
    CONSTRAINT ck_ingest_source_files_status CHECK ((status = ANY (ARRAY['pending'::text, 'claimed'::text, 'done'::text, 'failed'::text, 'skipped'::text])))
);

ALTER TABLE ONLY public.ingest_source_files FORCE ROW LEVEL SECURITY;


--
-- Name: ingest_sources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingest_sources (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    name text NOT NULL,
    kind text NOT NULL,
    config jsonb NOT NULL,
    schedule jsonb NOT NULL,
    enabled boolean DEFAULT true NOT NULL,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    last_poll_at timestamp with time zone,
    scan_state jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE ONLY public.ingest_sources FORCE ROW LEVEL SECURITY;


--
-- Name: job_attempts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_attempts (
    job_id uuid NOT NULL,
    attempt integer NOT NULL,
    fencing_token bigint NOT NULL,
    owner text NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    ended_at timestamp with time zone,
    outcome text,
    error_code text,
    error_message text,
    CONSTRAINT ck_job_attempts_outcome CHECK ((outcome = ANY (ARRAY['completed'::text, 'completed_with_errors'::text, 'failed'::text, 'cancelled'::text, 'paused'::text, 'interrupted'::text, 'lost'::text, 'time_limit'::text, 'yielded'::text])))
);

ALTER TABLE ONLY public.job_attempts FORCE ROW LEVEL SECURITY;


--
-- Name: job_items; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_items (
    job_id uuid NOT NULL,
    "position" integer NOT NULL,
    item_id text NOT NULL
);

ALTER TABLE ONLY public.job_items FORCE ROW LEVEL SECURITY;


--
-- Name: jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.jobs (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid,
    kind text NOT NULL,
    scope text DEFAULT ''::text NOT NULL,
    exclusive_group text,
    queue text NOT NULL,
    status text NOT NULL,
    params jsonb DEFAULT '{}'::jsonb NOT NULL,
    checkpoint jsonb,
    progress jsonb DEFAULT '{}'::jsonb NOT NULL,
    progress_updated_at timestamp with time zone,
    result jsonb,
    error_code text,
    error_message text,
    attempt integer DEFAULT 0 NOT NULL,
    retry_count integer DEFAULT 0 NOT NULL,
    max_attempts integer DEFAULT 3 NOT NULL,
    auto_resume boolean DEFAULT false NOT NULL,
    fencing_token bigint DEFAULT '0'::bigint NOT NULL,
    owner text,
    heartbeat_at timestamp with time zone,
    lease_expires_at timestamp with time zone,
    max_queue_wait interval NOT NULL,
    dispatch_count integer DEFAULT 0 NOT NULL,
    last_dispatched_at timestamp with time zone,
    cancel_requested_at timestamp with time zone,
    cancel_requested_by text,
    pause_requested_at timestamp with time zone,
    pause_reason text,
    idempotency_key text,
    parent_job_id uuid,
    requested_by text NOT NULL,
    requested_by_user_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    queued_at timestamp with time zone,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_jobs_finished_iff_terminal CHECK (((status = ANY (ARRAY['cancelled'::text, 'completed'::text, 'completed_with_errors'::text, 'failed'::text])) = (finished_at IS NOT NULL))),
    CONSTRAINT ck_jobs_idempotency_key CHECK (((length(idempotency_key) >= 8) AND (length(idempotency_key) <= 200))),
    CONSTRAINT ck_jobs_live_has_lease CHECK (((status <> ALL (ARRAY['running'::text, 'cancelling'::text])) OR ((owner IS NOT NULL) AND (lease_expires_at IS NOT NULL)))),
    CONSTRAINT ck_jobs_max_attempts CHECK (((max_attempts >= 1) AND (max_attempts <= 10))),
    CONSTRAINT ck_jobs_pause_reason CHECK ((pause_reason = ANY (ARRAY['user'::text, 'audit'::text, 'budget'::text, 'project_paused'::text, 'vlm_unavailable'::text]))),
    CONSTRAINT ck_jobs_paused_has_reason CHECK (((status <> 'paused'::text) OR (pause_reason IS NOT NULL))),
    CONSTRAINT ck_jobs_status CHECK ((status = ANY (ARRAY['queued'::text, 'running'::text, 'paused'::text, 'cancelling'::text, 'completed'::text, 'completed_with_errors'::text, 'failed'::text, 'cancelled'::text, 'interrupted'::text])))
)
WITH (fillfactor='70', autovacuum_vacuum_scale_factor='0.02', autovacuum_vacuum_threshold='50', autovacuum_vacuum_cost_delay='2');

ALTER TABLE ONLY public.jobs FORCE ROW LEVEL SECURITY;


--
-- Name: label_operation_members; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.label_operation_members (
    op_id uuid NOT NULL,
    item_id text NOT NULL,
    prior jsonb NOT NULL
);

ALTER TABLE ONLY public.label_operation_members FORCE ROW LEVEL SECURITY;


--
-- Name: label_operations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.label_operations (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    kind text NOT NULL,
    status text NOT NULL,
    params jsonb NOT NULL,
    member_count integer DEFAULT 0 NOT NULL,
    job_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    undone_at timestamp with time zone,
    CONSTRAINT ck_label_operations_status CHECK ((status = ANY (ARRAY['writing'::text, 'written'::text, 'undoing'::text, 'undone'::text, 'failed'::text])))
);

ALTER TABLE ONLY public.label_operations FORCE ROW LEVEL SECURITY;


--
-- Name: model_engines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_engines (
    id uuid DEFAULT uuidv7() NOT NULL,
    model_id uuid NOT NULL,
    triton_model_name text NOT NULL,
    "precision" text NOT NULL,
    input_size integer NOT NULL,
    input_contract jsonb NOT NULL,
    output_contract jsonb NOT NULL,
    build_profile text DEFAULT 'default'::text NOT NULL,
    build_settings jsonb NOT NULL,
    build_stamp text NOT NULL,
    onnx_sha256 text,
    engine_sha256 text,
    device_mib_measured integer,
    max_batch integer NOT NULL,
    opt_batch integer,
    cuda_graphs boolean DEFAULT false NOT NULL,
    parity_ref text,
    parity_report_key text,
    trt_version text NOT NULL,
    gpu_arch text NOT NULL,
    engine_storage_key text,
    build_job_id uuid,
    status text NOT NULL,
    built_at timestamp with time zone,
    CONSTRAINT ck_model_engines_build_profile CHECK ((build_profile = ANY (ARRAY['default'::text, 'small'::text]))),
    CONSTRAINT ck_model_engines_precision CHECK (("precision" = ANY (ARRAY['fp32'::text, 'fp16'::text, 'int8'::text]))),
    CONSTRAINT ck_model_engines_status CHECK ((status = ANY (ARRAY['building'::text, 'ready'::text, 'failed'::text])))
);

ALTER TABLE ONLY public.model_engines FORCE ROW LEVEL SECURITY;


--
-- Name: model_shares; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_shares (
    model_id uuid NOT NULL,
    consumer_project_id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL
);

ALTER TABLE ONLY public.model_shares FORCE ROW LEVEL SECURITY;


--
-- Name: models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.models (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid,
    name text NOT NULL,
    version integer NOT NULL,
    origin text NOT NULL,
    training_run_id uuid,
    family text NOT NULL,
    task text NOT NULL,
    labels text[] NOT NULL,
    artifact_storage_key text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    sharing_revision integer DEFAULT 1 NOT NULL,
    CONSTRAINT ck_models_origin CHECK ((origin = ANY (ARRAY['trained'::text, 'uploaded'::text, 'builtin'::text]))),
    CONSTRAINT ck_models_sharing_revision CHECK ((sharing_revision >= 1)),
    CONSTRAINT ck_models_version CHECK ((version >= 1))
);

ALTER TABLE ONLY public.models FORCE ROW LEVEL SECURITY;


--
-- Name: near_dup_decisions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.near_dup_decisions (
    id uuid DEFAULT uuidv7() NOT NULL,
    level text NOT NULL,
    a_id text NOT NULL,
    b_id text NOT NULL,
    decision text NOT NULL,
    decided_by text NOT NULL,
    decided_by_user_id uuid,
    decided_at timestamp with time zone DEFAULT now() NOT NULL,
    project_id uuid NOT NULL,
    CONSTRAINT ck_near_dup_decisions_decision CHECK ((decision = ANY (ARRAY['same'::text, 'different'::text]))),
    CONSTRAINT ck_near_dup_decisions_level CHECK ((level = ANY (ARRAY['image'::text, 'item'::text]))),
    CONSTRAINT ck_near_dup_decisions_ordered CHECK ((a_id < b_id))
);

ALTER TABLE ONLY public.near_dup_decisions FORCE ROW LEVEL SECURITY;


--
-- Name: near_dup_edges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.near_dup_edges (
    id bigint NOT NULL,
    level text NOT NULL,
    space text NOT NULL,
    a_id text NOT NULL,
    b_id text NOT NULL,
    kind text NOT NULL,
    score real NOT NULL,
    computed_at timestamp with time zone DEFAULT now() NOT NULL,
    project_id uuid,
    CONSTRAINT ck_near_dup_edges_kind CHECK ((kind = 'embedding'::text)),
    CONSTRAINT ck_near_dup_edges_level CHECK ((level = ANY (ARRAY['image'::text, 'item'::text, 'core'::text]))),
    CONSTRAINT ck_near_dup_edges_ordered CHECK ((a_id < b_id))
);

ALTER TABLE ONLY public.near_dup_edges FORCE ROW LEVEL SECURITY;


--
-- Name: near_dup_edges_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_edges ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.near_dup_edges_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: near_dup_groups; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.near_dup_groups (
    id uuid DEFAULT uuidv7() NOT NULL,
    purpose text NOT NULL,
    lineage_id uuid NOT NULL,
    representative_id text NOT NULL,
    member_count integer NOT NULL,
    min_edge_cosine real NOT NULL,
    min_cosine_to_rep real NOT NULL,
    has_holdout boolean DEFAULT false NOT NULL,
    run_id uuid NOT NULL,
    project_id uuid,
    CONSTRAINT ck_near_dup_groups_purpose CHECK ((purpose = ANY (ARRAY['dedup'::text, 'leak'::text]))),
    CONSTRAINT ck_near_dup_groups_size CHECK ((member_count >= 2))
);

ALTER TABLE ONLY public.near_dup_groups FORCE ROW LEVEL SECURITY;


--
-- Name: near_dup_members; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.near_dup_members (
    purpose text NOT NULL,
    member_id text NOT NULL,
    group_id uuid NOT NULL,
    is_representative boolean NOT NULL,
    cosine_to_rep real NOT NULL,
    run_id uuid NOT NULL,
    project_id uuid,
    CONSTRAINT ck_near_dup_members_purpose CHECK ((purpose = ANY (ARRAY['dedup'::text, 'leak'::text])))
);

ALTER TABLE ONLY public.near_dup_members FORCE ROW LEVEL SECURITY;


--
-- Name: near_dup_pending; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.near_dup_pending (
    id bigint NOT NULL,
    level text NOT NULL,
    member_id text NOT NULL,
    reason text NOT NULL,
    enqueued_at timestamp with time zone DEFAULT now() NOT NULL,
    claimed_by_job_id uuid,
    project_id uuid,
    CONSTRAINT ck_near_dup_pending_level CHECK ((level = ANY (ARRAY['image'::text, 'item'::text, 'core'::text]))),
    CONSTRAINT ck_near_dup_pending_reason CHECK ((reason = ANY (ARRAY['added'::text, 'vector_changed'::text, 'boxes_changed'::text, 'deleted'::text, 'group_changed'::text])))
);

ALTER TABLE ONLY public.near_dup_pending FORCE ROW LEVEL SECURITY;


--
-- Name: near_dup_pending_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_pending ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.near_dup_pending_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: near_dup_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.near_dup_runs (
    id uuid DEFAULT uuidv7() NOT NULL,
    level text NOT NULL,
    mode text NOT NULL,
    status text NOT NULL,
    job_id uuid,
    parent_run_id uuid,
    profile_config_id uuid,
    profile_revision integer,
    params jsonb NOT NULL,
    space text NOT NULL,
    algorithm_version text NOT NULL,
    members_scored integer DEFAULT 0 NOT NULL,
    dedup_groups integer DEFAULT 0 NOT NULL,
    leak_groups integer DEFAULT 0 NOT NULL,
    grouped_members integer DEFAULT 0 NOT NULL,
    largest_group integer DEFAULT 0 NOT NULL,
    oversized_split integer DEFAULT 0 NOT NULL,
    holdout_absorbed integer DEFAULT 0 NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    project_id uuid,
    CONSTRAINT ck_near_dup_runs_building_unfinished CHECK (((status = 'building'::text) = (finished_at IS NULL))),
    CONSTRAINT ck_near_dup_runs_core_has_no_profile CHECK (((level = 'core'::text) = (profile_config_id IS NULL))),
    CONSTRAINT ck_near_dup_runs_core_has_no_project CHECK (((level = 'core'::text) = (project_id IS NULL))),
    CONSTRAINT ck_near_dup_runs_level CHECK ((level = ANY (ARRAY['image'::text, 'item'::text, 'core'::text]))),
    CONSTRAINT ck_near_dup_runs_mode CHECK ((mode = ANY (ARRAY['full'::text, 'incremental'::text]))),
    CONSTRAINT ck_near_dup_runs_profile_pair CHECK (((profile_config_id IS NULL) = (profile_revision IS NULL))),
    CONSTRAINT ck_near_dup_runs_status CHECK ((status = ANY (ARRAY['building'::text, 'current'::text, 'superseded'::text, 'failed'::text])))
);

ALTER TABLE ONLY public.near_dup_runs FORCE ROW LEVEL SECURITY;


--
-- Name: partition_clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.partition_clusters (
    partition_id uuid NOT NULL,
    cluster_uid uuid NOT NULL,
    project_id uuid,
    display_no integer NOT NULL,
    parent_uid uuid,
    transition text NOT NULL,
    predecessor_uids uuid[] DEFAULT '{}'::uuid[] NOT NULL,
    size integer NOT NULL,
    dispersion real,
    prototypes jsonb DEFAULT '[]'::jsonb NOT NULL,
    CONSTRAINT ck_partition_clusters_size CHECK ((size > 0)),
    CONSTRAINT ck_partition_clusters_transition CHECK ((transition = ANY (ARRAY['new'::text, 'survived'::text, 'split'::text, 'merged'::text, 'absorbed'::text])))
);

ALTER TABLE ONLY public.partition_clusters FORCE ROW LEVEL SECURITY;


--
-- Name: privacy_purges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.privacy_purges (
    id uuid DEFAULT uuidv7() NOT NULL,
    kind text NOT NULL,
    at timestamp with time zone DEFAULT now() NOT NULL,
    counts jsonb NOT NULL,
    actor text NOT NULL,
    host text NOT NULL,
    CONSTRAINT ck_privacy_purges_counts_only CHECK (((jsonb_typeof(counts) = 'object'::text) AND (NOT jsonb_path_exists(counts, '$.*?(@.type() != "number" && @.type() != "boolean")'::jsonpath)))),
    CONSTRAINT ck_privacy_purges_kind CHECK ((kind = ANY (ARRAY['faces'::text, 'ocr'::text])))
);


--
-- Name: project_classes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.project_classes (
    project_id uuid NOT NULL,
    class_id integer NOT NULL,
    name text NOT NULL,
    class_group text,
    deprecated boolean DEFAULT false NOT NULL,
    merged_into integer,
    hotkey_letter text,
    notes text DEFAULT ''::text NOT NULL,
    added_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    name_key text NOT NULL,
    CONSTRAINT ck_project_classes_class_id CHECK ((class_id >= 0)),
    CONSTRAINT ck_project_classes_hotkey_letter CHECK ((hotkey_letter ~ '^[a-z0-9]$'::text)),
    CONSTRAINT ck_project_classes_merged_into_deprecated CHECK (((merged_into IS NULL) OR deprecated)),
    CONSTRAINT ck_project_classes_merged_into_other CHECK (((merged_into IS NULL) OR (merged_into <> class_id))),
    CONSTRAINT ck_project_classes_name CHECK ((((length(name) >= 1) AND (length(name) <= 200)) AND (name = btrim(name)))),
    CONSTRAINT ck_project_classes_name_key CHECK ((name_key ~ '^[a-z0-9_]+$'::text))
);

ALTER TABLE ONLY public.project_classes FORCE ROW LEVEL SECURITY;


--
-- Name: project_indexes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.project_indexes (
    project_id uuid NOT NULL,
    role text NOT NULL,
    index_name text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_project_indexes_role CHECK ((role = ANY (ARRAY['images'::text, 'items'::text])))
);

ALTER TABLE ONLY public.project_indexes FORCE ROW LEVEL SECURITY;


--
-- Name: project_keymaps; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.project_keymaps (
    project_id uuid NOT NULL,
    overrides jsonb DEFAULT '{}'::jsonb NOT NULL,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_by text NOT NULL
);

ALTER TABLE ONLY public.project_keymaps FORCE ROW LEVEL SECURITY;


--
-- Name: project_settings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.project_settings (
    project_id uuid NOT NULL,
    defaults jsonb DEFAULT '{}'::jsonb NOT NULL,
    ingest_policy jsonb,
    vlm_policy jsonb,
    vlm_scope_reps jsonb,
    search jsonb,
    location_policy text DEFAULT 'store'::text NOT NULL,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_by text NOT NULL,
    CONSTRAINT ck_project_settings_location_policy CHECK ((location_policy = ANY (ARRAY['drop'::text, 'store_coarse'::text, 'store'::text])))
);

ALTER TABLE ONLY public.project_settings FORCE ROW LEVEL SECURITY;


--
-- Name: project_slugs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.project_slugs (
    slug text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    retired_at timestamp with time zone,
    CONSTRAINT ck_project_slugs_slug CHECK (((slug ~ '^[a-z][a-z0-9-]{1,31}$'::text) AND (slug !~ '--'::text) AND (slug !~ '-$'::text)))
);


--
-- Name: project_state; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.project_state (
    project_id uuid NOT NULL,
    key text NOT NULL,
    value jsonb NOT NULL,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_project_state_key CHECK ((key = ANY (ARRAY['cluster_refresh'::text, 'region_drain'::text, 'region_dependency_health'::text, 'vlm_external_ack'::text, 'open_vocab_gate'::text, 'region_text_consensus_marker'::text])))
);

ALTER TABLE ONLY public.project_state FORCE ROW LEVEL SECURITY;


--
-- Name: projection_states; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.projection_states (
    project_id uuid NOT NULL,
    projection_version integer NOT NULL,
    scope text NOT NULL,
    cluster_uid uuid,
    n_points integer NOT NULL,
    n_components integer NOT NULL,
    metric text NOT NULL,
    fitted_at timestamp with time zone NOT NULL,
    reducer_storage_key text,
    job_id uuid
);

ALTER TABLE ONLY public.projection_states FORCE ROW LEVEL SECURITY;


--
-- Name: projects; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.projects (
    id uuid DEFAULT uuidv7() NOT NULL,
    lock_key integer NOT NULL,
    slug text NOT NULL,
    display_name text NOT NULL,
    description text DEFAULT ''::text NOT NULL,
    status text NOT NULL,
    pre_delete_status text,
    revision bigint DEFAULT '1'::bigint NOT NULL,
    config_epoch bigint DEFAULT '0'::bigint NOT NULL,
    paused_at timestamp with time zone,
    paused_by text,
    origin jsonb,
    index_prefix text NOT NULL,
    storage_prefix text NOT NULL,
    model_prefix text NOT NULL,
    mlflow_experiment text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by text NOT NULL,
    created_by_user_id uuid,
    CONSTRAINT ck_projects_deleting_has_prior_status CHECK (((status = 'deleting'::text) = (pre_delete_status IS NOT NULL))),
    CONSTRAINT ck_projects_pre_delete_status CHECK ((pre_delete_status = ANY (ARRAY['active'::text, 'archived'::text, 'failed'::text]))),
    CONSTRAINT ck_projects_slug_reserved CHECK ((NOT (slug = ANY (ARRAY['combine'::text, 'new'::text, 'all'::text, 'none'::text, 'projects'::text, 'global'::text, 'settings'::text, 'vlm'::text, 'health'::text])))),
    CONSTRAINT ck_projects_status CHECK ((status = ANY (ARRAY['building'::text, 'active'::text, 'archived'::text, 'deleting'::text, 'failed'::text])))
);


--
-- Name: projects_lock_key_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.projects ALTER COLUMN lock_key ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.projects_lock_key_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: promotions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.promotions (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    model_id uuid NOT NULL,
    triton_model_name text NOT NULL,
    triton_version integer NOT NULL,
    class_remap jsonb,
    job_id uuid,
    promoted_at timestamp with time zone DEFAULT now() NOT NULL,
    promoted_by text NOT NULL,
    superseded_at timestamp with time zone
);

ALTER TABLE ONLY public.promotions FORCE ROW LEVEL SECURITY;


--
-- Name: region_text_consensus_groups; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.region_text_consensus_groups (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    run_id uuid NOT NULL,
    basis text NOT NULL,
    method text NOT NULL,
    n_frames integer NOT NULL,
    n_agree integer NOT NULL,
    confidence real,
    tier text NOT NULL,
    jurisdiction text,
    jurisdiction_conf real,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_region_text_consensus_groups_basis CHECK ((basis = ANY (ARRAY['near_dup'::text, 'capture_window'::text, 'human'::text]))),
    CONSTRAINT ck_region_text_consensus_groups_jurisdiction_conf CHECK (((jurisdiction_conf >= (0)::double precision) AND (jurisdiction_conf <= (1)::double precision))),
    CONSTRAINT ck_region_text_consensus_groups_method CHECK ((method = ANY (ARRAY['best_frame'::text, 'position_vote'::text, 'aligned_vote'::text]))),
    CONSTRAINT ck_region_text_consensus_groups_n_agree CHECK (((n_agree >= 0) AND (n_agree <= n_frames))),
    CONSTRAINT ck_region_text_consensus_groups_n_frames CHECK ((n_frames >= 1)),
    CONSTRAINT ck_region_text_consensus_groups_tier CHECK ((tier = ANY (ARRAY['confident'::text, 'partial'::text, 'none'::text])))
);

ALTER TABLE ONLY public.region_text_consensus_groups FORCE ROW LEVEL SECURITY;


--
-- Name: region_text_consensus_members; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.region_text_consensus_members (
    group_id uuid NOT NULL,
    item_id text NOT NULL,
    box_id text NOT NULL,
    project_id uuid NOT NULL,
    weight real NOT NULL,
    agrees boolean NOT NULL
);

ALTER TABLE ONLY public.region_text_consensus_members FORCE ROW LEVEL SECURITY;


--
-- Name: region_text_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.region_text_runs (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    job_id uuid,
    trigger text NOT NULL,
    reader text NOT NULL,
    reader_version text NOT NULL,
    profile_config_id uuid,
    profile_revision integer,
    params jsonb DEFAULT '{}'::jsonb NOT NULL,
    counts jsonb DEFAULT '{}'::jsonb NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    created_by text NOT NULL,
    CONSTRAINT ck_region_text_runs_profile_pair CHECK (((profile_config_id IS NULL) = (profile_revision IS NULL))),
    CONSTRAINT ck_region_text_runs_trigger CHECK ((trigger = ANY (ARRAY['region_pass'::text, 'reread'::text, 'consensus'::text])))
);

ALTER TABLE ONLY public.region_text_runs FORCE ROW LEVEL SECURITY;


--
-- Name: search_projection_tasks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.search_projection_tasks (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    kind text NOT NULL,
    payload jsonb NOT NULL,
    job_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    done_at timestamp with time zone,
    CONSTRAINT ck_search_projection_tasks_kind CHECK ((kind = ANY (ARRAY['class_rename'::text, 'class_merge'::text, 'cluster_partition'::text, 'near_dup_run'::text])))
);

ALTER TABLE ONLY public.search_projection_tasks FORCE ROW LEVEL SECURITY;


--
-- Name: serving_reports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.serving_reports (
    server text NOT NULL,
    model text NOT NULL,
    version integer NOT NULL,
    state text NOT NULL,
    instances integer,
    device_mib integer,
    seen_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT ck_serving_reports_state CHECK ((state = ANY (ARRAY['ready'::text, 'unavailable'::text, 'loading'::text, 'unloading'::text])))
);


--
-- Name: stream_worker_claims; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stream_worker_claims (
    worker_kind text NOT NULL,
    project_id uuid NOT NULL,
    worker_id text NOT NULL,
    fencing_token bigint DEFAULT '1'::bigint NOT NULL,
    lease_expires_at timestamp with time zone NOT NULL,
    CONSTRAINT ck_stream_worker_claims_worker_kind CHECK ((worker_kind = ANY (ARRAY['detection'::text, 'vlm'::text])))
);

ALTER TABLE ONLY public.stream_worker_claims FORCE ROW LEVEL SECURITY;


--
-- Name: training_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.training_runs (
    id uuid DEFAULT uuidv7() NOT NULL,
    project_id uuid NOT NULL,
    job_id uuid NOT NULL,
    dataset_export_id uuid,
    spec jsonb NOT NULL,
    class_names text[] NOT NULL,
    family text NOT NULL,
    base_model text NOT NULL,
    artifact_prefix text NOT NULL,
    manifest jsonb,
    mlflow_run_id text,
    gpu_uuids text[] DEFAULT '{}'::text[] NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone
);

ALTER TABLE ONLY public.training_runs FORCE ROW LEVEL SECURITY;


--
-- Name: upgrade_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.upgrade_runs (
    id uuid DEFAULT uuidv7() NOT NULL,
    from_version text NOT NULL,
    to_version text NOT NULL,
    step text NOT NULL,
    status text NOT NULL,
    source_fingerprint text NOT NULL,
    report jsonb,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    CONSTRAINT ck_upgrade_runs_status CHECK ((status = ANY (ARRAY['running'::text, 'completed'::text, 'failed'::text]))),
    CONSTRAINT ck_upgrade_runs_step CHECK ((step = ANY (ARRAY['export_bundle'::text, 'load'::text, 'verify'::text, 'delete_sources'::text])))
);


--
-- Name: vlm_endpoint_probes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.vlm_endpoint_probes (
    config_id uuid NOT NULL,
    revision integer NOT NULL,
    probed_at timestamp with time zone DEFAULT now() NOT NULL,
    ok boolean NOT NULL,
    result jsonb NOT NULL
);

ALTER TABLE ONLY public.vlm_endpoint_probes FORCE ROW LEVEL SECURITY;


--
-- Name: vlm_usage_hourly; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.vlm_usage_hourly (
    project_id uuid NOT NULL,
    endpoint_config_id uuid NOT NULL,
    source text NOT NULL,
    hour timestamp with time zone NOT NULL,
    requests bigint DEFAULT '0'::bigint NOT NULL,
    crops bigint DEFAULT '0'::bigint NOT NULL,
    prompt_tokens bigint DEFAULT '0'::bigint NOT NULL,
    completion_tokens bigint DEFAULT '0'::bigint NOT NULL,
    usage_missing bigint DEFAULT '0'::bigint NOT NULL,
    CONSTRAINT ck_vlm_usage_hourly_hour CHECK ((hour = date_trunc('hour'::text, hour))),
    CONSTRAINT ck_vlm_usage_hourly_source CHECK ((source = ANY (ARRAY['assist_loop'::text, 'assist_run'::text, 'autolabel'::text, 'vlm_worker'::text, 'region_verify'::text, 'label_batch'::text, 'open_vocab_gate'::text])))
);

ALTER TABLE ONLY public.vlm_usage_hourly FORCE ROW LEVEL SECURITY;


--
-- Name: worker_capabilities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.worker_capabilities (
    worker_id text NOT NULL,
    queue text NOT NULL,
    host text NOT NULL,
    capabilities jsonb NOT NULL,
    seen_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: assist_loop_budget_ledger pk_assist_loop_budget_ledger; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_budget_ledger
    ADD CONSTRAINT pk_assist_loop_budget_ledger PRIMARY KEY (id);


--
-- Name: assist_loop_budgets pk_assist_loop_budgets; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_budgets
    ADD CONSTRAINT pk_assist_loop_budgets PRIMARY KEY (run_id, pool, unit);


--
-- Name: assist_loop_iterations pk_assist_loop_iterations; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_iterations
    ADD CONSTRAINT pk_assist_loop_iterations PRIMARY KEY (run_id, ordinal);


--
-- Name: assist_loop_nodes pk_assist_loop_nodes; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_nodes
    ADD CONSTRAINT pk_assist_loop_nodes PRIMARY KEY (run_id, path);


--
-- Name: assist_loop_proposals pk_assist_loop_proposals; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_proposals
    ADD CONSTRAINT pk_assist_loop_proposals PRIMARY KEY (run_id, item_id, role);


--
-- Name: assist_loop_reviews pk_assist_loop_reviews; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_reviews
    ADD CONSTRAINT pk_assist_loop_reviews PRIMARY KEY (id);


--
-- Name: assist_loop_runs pk_assist_loop_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_runs
    ADD CONSTRAINT pk_assist_loop_runs PRIMARY KEY (id);


--
-- Name: assist_loop_selections pk_assist_loop_selections; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_selections
    ADD CONSTRAINT pk_assist_loop_selections PRIMARY KEY (run_id, item_id, purpose);


--
-- Name: bakeoff_runs pk_bakeoff_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bakeoff_runs
    ADD CONSTRAINT pk_bakeoff_runs PRIMARY KEY (job_id);


--
-- Name: class_registries pk_class_registries; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.class_registries
    ADD CONSTRAINT pk_class_registries PRIMARY KEY (project_id);


--
-- Name: cluster_activation_events pk_cluster_activation_events; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_activation_events
    ADD CONSTRAINT pk_cluster_activation_events PRIMARY KEY (id);


--
-- Name: cluster_activations pk_cluster_activations; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_activations
    ADD CONSTRAINT pk_cluster_activations PRIMARY KEY (id);


--
-- Name: cluster_backend_validations pk_cluster_backend_validations; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_backend_validations
    ADD CONSTRAINT pk_cluster_backend_validations PRIMARY KEY (id);


--
-- Name: cluster_calibrations pk_cluster_calibrations; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_calibrations
    ADD CONSTRAINT pk_cluster_calibrations PRIMARY KEY (id);


--
-- Name: cluster_identities pk_cluster_identities; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_identities
    ADD CONSTRAINT pk_cluster_identities PRIMARY KEY (cluster_uid);


--
-- Name: cluster_memberships pk_cluster_memberships; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_memberships
    ADD CONSTRAINT pk_cluster_memberships PRIMARY KEY (partition_id, item_key);


--
-- Name: cluster_partitions pk_cluster_partitions; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_partitions
    ADD CONSTRAINT pk_cluster_partitions PRIMARY KEY (id);


--
-- Name: combine_runs pk_combine_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine_runs
    ADD CONSTRAINT pk_combine_runs PRIMARY KEY (job_id);


--
-- Name: combine_sources pk_combine_sources; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine_sources
    ADD CONSTRAINT pk_combine_sources PRIMARY KEY (job_id, "position");


--
-- Name: config_activation_events pk_config_activation_events; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_activation_events
    ADD CONSTRAINT pk_config_activation_events PRIMARY KEY (id);


--
-- Name: config_activations pk_config_activations; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_activations
    ADD CONSTRAINT pk_config_activations PRIMARY KEY (project_id, axis);


--
-- Name: config_documents pk_config_documents; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_documents
    ADD CONSTRAINT pk_config_documents PRIMARY KEY (id);


--
-- Name: config_revisions pk_config_revisions; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_revisions
    ADD CONSTRAINT pk_config_revisions PRIMARY KEY (config_id, revision);


--
-- Name: config_runtime_reports pk_config_runtime_reports; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_runtime_reports
    ADD CONSTRAINT pk_config_runtime_reports PRIMARY KEY (project_id, process, host);


--
-- Name: dataset_exports pk_dataset_exports; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_exports
    ADD CONSTRAINT pk_dataset_exports PRIMARY KEY (id);


--
-- Name: dataset_import_images pk_dataset_import_images; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_import_images
    ADD CONSTRAINT pk_dataset_import_images PRIMARY KEY (import_id, image_key);


--
-- Name: dataset_imports pk_dataset_imports; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_imports
    ADD CONSTRAINT pk_dataset_imports PRIMARY KEY (id);


--
-- Name: dataset_uploads pk_dataset_uploads; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_uploads
    ADD CONSTRAINT pk_dataset_uploads PRIMARY KEY (id);


--
-- Name: deployment_counters pk_deployment_counters; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.deployment_counters
    ADD CONSTRAINT pk_deployment_counters PRIMARY KEY (name);


--
-- Name: deployment_settings pk_deployment_settings; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.deployment_settings
    ADD CONSTRAINT pk_deployment_settings PRIMARY KEY (key);


--
-- Name: embedding_evaluation_runs pk_embedding_evaluation_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_evaluation_runs
    ADD CONSTRAINT pk_embedding_evaluation_runs PRIMARY KEY (id);


--
-- Name: embedding_model_versions pk_embedding_model_versions; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_model_versions
    ADD CONSTRAINT pk_embedding_model_versions PRIMARY KEY (id);


--
-- Name: embedding_reindex_runs pk_embedding_reindex_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_reindex_runs
    ADD CONSTRAINT pk_embedding_reindex_runs PRIMARY KEY (id);


--
-- Name: embedding_spaces pk_embedding_spaces; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_spaces
    ADD CONSTRAINT pk_embedding_spaces PRIMARY KEY (id);


--
-- Name: events pk_events; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT pk_events PRIMARY KEY (id);


--
-- Name: gpu_devices pk_gpu_devices; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_devices
    ADD CONSTRAINT pk_gpu_devices PRIMARY KEY (gpu_uuid);


--
-- Name: gpu_reservations pk_gpu_reservations; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_reservations
    ADD CONSTRAINT pk_gpu_reservations PRIMARY KEY (gpu_uuid);


--
-- Name: holdout_freezes pk_holdout_freezes; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_freezes
    ADD CONSTRAINT pk_holdout_freezes PRIMARY KEY (id);


--
-- Name: holdout_near_dup_moves pk_holdout_near_dup_moves; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_near_dup_moves
    ADD CONSTRAINT pk_holdout_near_dup_moves PRIMARY KEY (id);


--
-- Name: image_contents pk_image_contents; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.image_contents
    ADD CONSTRAINT pk_image_contents PRIMARY KEY (id);


--
-- Name: ingest_source_files pk_ingest_source_files; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingest_source_files
    ADD CONSTRAINT pk_ingest_source_files PRIMARY KEY (source_id, file_key);


--
-- Name: ingest_sources pk_ingest_sources; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingest_sources
    ADD CONSTRAINT pk_ingest_sources PRIMARY KEY (id);


--
-- Name: job_attempts pk_job_attempts; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_attempts
    ADD CONSTRAINT pk_job_attempts PRIMARY KEY (job_id, attempt);


--
-- Name: job_items pk_job_items; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_items
    ADD CONSTRAINT pk_job_items PRIMARY KEY (job_id, "position");


--
-- Name: jobs pk_jobs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT pk_jobs PRIMARY KEY (id);


--
-- Name: label_operation_members pk_label_operation_members; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.label_operation_members
    ADD CONSTRAINT pk_label_operation_members PRIMARY KEY (op_id, item_id);


--
-- Name: label_operations pk_label_operations; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.label_operations
    ADD CONSTRAINT pk_label_operations PRIMARY KEY (id);


--
-- Name: model_engines pk_model_engines; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_engines
    ADD CONSTRAINT pk_model_engines PRIMARY KEY (id);


--
-- Name: model_shares pk_model_shares; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_shares
    ADD CONSTRAINT pk_model_shares PRIMARY KEY (model_id, consumer_project_id);


--
-- Name: models pk_models; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT pk_models PRIMARY KEY (id);


--
-- Name: near_dup_decisions pk_near_dup_decisions; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_decisions
    ADD CONSTRAINT pk_near_dup_decisions PRIMARY KEY (id);


--
-- Name: near_dup_edges pk_near_dup_edges; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_edges
    ADD CONSTRAINT pk_near_dup_edges PRIMARY KEY (id);


--
-- Name: near_dup_groups pk_near_dup_groups; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_groups
    ADD CONSTRAINT pk_near_dup_groups PRIMARY KEY (id);


--
-- Name: near_dup_members pk_near_dup_members; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_members
    ADD CONSTRAINT pk_near_dup_members PRIMARY KEY (run_id, purpose, member_id);


--
-- Name: near_dup_pending pk_near_dup_pending; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_pending
    ADD CONSTRAINT pk_near_dup_pending PRIMARY KEY (id);


--
-- Name: near_dup_runs pk_near_dup_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_runs
    ADD CONSTRAINT pk_near_dup_runs PRIMARY KEY (id);


--
-- Name: partition_clusters pk_partition_clusters; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.partition_clusters
    ADD CONSTRAINT pk_partition_clusters PRIMARY KEY (partition_id, cluster_uid);


--
-- Name: privacy_purges pk_privacy_purges; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.privacy_purges
    ADD CONSTRAINT pk_privacy_purges PRIMARY KEY (id);


--
-- Name: project_classes pk_project_classes; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_classes
    ADD CONSTRAINT pk_project_classes PRIMARY KEY (project_id, class_id);


--
-- Name: project_indexes pk_project_indexes; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_indexes
    ADD CONSTRAINT pk_project_indexes PRIMARY KEY (project_id, role);


--
-- Name: project_keymaps pk_project_keymaps; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_keymaps
    ADD CONSTRAINT pk_project_keymaps PRIMARY KEY (project_id);


--
-- Name: project_settings pk_project_settings; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_settings
    ADD CONSTRAINT pk_project_settings PRIMARY KEY (project_id);


--
-- Name: project_slugs pk_project_slugs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_slugs
    ADD CONSTRAINT pk_project_slugs PRIMARY KEY (slug);


--
-- Name: project_state pk_project_state; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_state
    ADD CONSTRAINT pk_project_state PRIMARY KEY (project_id, key);


--
-- Name: projection_states pk_projection_states; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.projection_states
    ADD CONSTRAINT pk_projection_states PRIMARY KEY (project_id);


--
-- Name: projects pk_projects; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.projects
    ADD CONSTRAINT pk_projects PRIMARY KEY (id);


--
-- Name: promotions pk_promotions; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.promotions
    ADD CONSTRAINT pk_promotions PRIMARY KEY (id);


--
-- Name: region_text_consensus_groups pk_region_text_consensus_groups; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_groups
    ADD CONSTRAINT pk_region_text_consensus_groups PRIMARY KEY (id);


--
-- Name: region_text_consensus_members pk_region_text_consensus_members; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_members
    ADD CONSTRAINT pk_region_text_consensus_members PRIMARY KEY (group_id, item_id, box_id);


--
-- Name: region_text_runs pk_region_text_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_runs
    ADD CONSTRAINT pk_region_text_runs PRIMARY KEY (id);


--
-- Name: search_projection_tasks pk_search_projection_tasks; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_projection_tasks
    ADD CONSTRAINT pk_search_projection_tasks PRIMARY KEY (id);


--
-- Name: serving_reports pk_serving_reports; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.serving_reports
    ADD CONSTRAINT pk_serving_reports PRIMARY KEY (server, model, version);


--
-- Name: stream_worker_claims pk_stream_worker_claims; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stream_worker_claims
    ADD CONSTRAINT pk_stream_worker_claims PRIMARY KEY (worker_kind, project_id);


--
-- Name: training_runs pk_training_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_runs
    ADD CONSTRAINT pk_training_runs PRIMARY KEY (id);


--
-- Name: upgrade_runs pk_upgrade_runs; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.upgrade_runs
    ADD CONSTRAINT pk_upgrade_runs PRIMARY KEY (id);


--
-- Name: vlm_endpoint_probes pk_vlm_endpoint_probes; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vlm_endpoint_probes
    ADD CONSTRAINT pk_vlm_endpoint_probes PRIMARY KEY (config_id, revision);


--
-- Name: vlm_usage_hourly pk_vlm_usage_hourly; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vlm_usage_hourly
    ADD CONSTRAINT pk_vlm_usage_hourly PRIMARY KEY (project_id, endpoint_config_id, source, hour);


--
-- Name: worker_capabilities pk_worker_capabilities; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worker_capabilities
    ADD CONSTRAINT pk_worker_capabilities PRIMARY KEY (worker_id);


--
-- Name: assist_loop_reviews uq_assist_loop_reviews_item_queue; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_reviews
    ADD CONSTRAINT uq_assist_loop_reviews_item_queue UNIQUE (run_id, item_id, queue);


--
-- Name: assist_loop_runs uq_assist_loop_runs_job_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_runs
    ADD CONSTRAINT uq_assist_loop_runs_job_id UNIQUE (job_id);


--
-- Name: cluster_activations uq_cluster_activations_served_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_activations
    ADD CONSTRAINT uq_cluster_activations_served_key UNIQUE NULLS NOT DISTINCT (project_id, target, level, embedding_field);


--
-- Name: cluster_partitions uq_cluster_partitions_id_project_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_partitions
    ADD CONSTRAINT uq_cluster_partitions_id_project_id UNIQUE (id, project_id);


--
-- Name: config_documents uq_config_documents_id_kind; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_documents
    ADD CONSTRAINT uq_config_documents_id_kind UNIQUE (id, kind);


--
-- Name: config_documents uq_config_documents_project_id_kind_name; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_documents
    ADD CONSTRAINT uq_config_documents_project_id_kind_name UNIQUE NULLS NOT DISTINCT (project_id, kind, name);


--
-- Name: dataset_imports uq_dataset_imports_job_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_imports
    ADD CONSTRAINT uq_dataset_imports_job_id UNIQUE (job_id);


--
-- Name: dataset_uploads uq_dataset_uploads_storage_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_uploads
    ADD CONSTRAINT uq_dataset_uploads_storage_key UNIQUE (storage_key);


--
-- Name: embedding_model_versions uq_embedding_model_versions_key_version; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_model_versions
    ADD CONSTRAINT uq_embedding_model_versions_key_version UNIQUE (model_key, version);


--
-- Name: embedding_reindex_runs uq_embedding_reindex_runs_job_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_reindex_runs
    ADD CONSTRAINT uq_embedding_reindex_runs_job_id UNIQUE (job_id);


--
-- Name: embedding_spaces uq_embedding_spaces_target_space_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_spaces
    ADD CONSTRAINT uq_embedding_spaces_target_space_key UNIQUE (project_id, target, space_key);


--
-- Name: embedding_spaces uq_embedding_spaces_vector_field; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_spaces
    ADD CONSTRAINT uq_embedding_spaces_vector_field UNIQUE (project_id, index_role, vector_field);


--
-- Name: holdout_near_dup_moves uq_holdout_near_dup_moves_item; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_near_dup_moves
    ADD CONSTRAINT uq_holdout_near_dup_moves_item UNIQUE (project_id, item_id);


--
-- Name: image_contents uq_image_contents_image; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.image_contents
    ADD CONSTRAINT uq_image_contents_image UNIQUE NULLS NOT DISTINCT (project_id, image_id);


--
-- Name: ingest_sources uq_ingest_sources_project_id_name; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingest_sources
    ADD CONSTRAINT uq_ingest_sources_project_id_name UNIQUE (project_id, name);


--
-- Name: job_items uq_job_items_job_id_item_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_items
    ADD CONSTRAINT uq_job_items_job_id_item_id UNIQUE (job_id, item_id);


--
-- Name: model_engines uq_model_engines_build_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_engines
    ADD CONSTRAINT uq_model_engines_build_key UNIQUE (triton_model_name, trt_version, gpu_arch, build_profile);


--
-- Name: models uq_models_project_id_name_version; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT uq_models_project_id_name_version UNIQUE NULLS NOT DISTINCT (project_id, name, version);


--
-- Name: near_dup_decisions uq_near_dup_decisions_pair; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_decisions
    ADD CONSTRAINT uq_near_dup_decisions_pair UNIQUE (project_id, level, a_id, b_id);


--
-- Name: near_dup_edges uq_near_dup_edges_pair; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_edges
    ADD CONSTRAINT uq_near_dup_edges_pair UNIQUE NULLS NOT DISTINCT (project_id, level, space, kind, a_id, b_id);


--
-- Name: near_dup_groups uq_near_dup_groups_id_project_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_groups
    ADD CONSTRAINT uq_near_dup_groups_id_project_id UNIQUE (id, project_id);


--
-- Name: near_dup_pending uq_near_dup_pending_member; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_pending
    ADD CONSTRAINT uq_near_dup_pending_member UNIQUE NULLS NOT DISTINCT (project_id, level, member_id);


--
-- Name: near_dup_runs uq_near_dup_runs_id_project_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_runs
    ADD CONSTRAINT uq_near_dup_runs_id_project_id UNIQUE (id, project_id);


--
-- Name: partition_clusters uq_partition_clusters_display_no; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.partition_clusters
    ADD CONSTRAINT uq_partition_clusters_display_no UNIQUE (partition_id, display_no);


--
-- Name: project_indexes uq_project_indexes_index_name; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_indexes
    ADD CONSTRAINT uq_project_indexes_index_name UNIQUE (index_name);


--
-- Name: projects uq_projects_lock_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.projects
    ADD CONSTRAINT uq_projects_lock_key UNIQUE (lock_key);


--
-- Name: projects uq_projects_slug; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.projects
    ADD CONSTRAINT uq_projects_slug UNIQUE (slug);


--
-- Name: region_text_consensus_groups uq_region_text_consensus_groups_id_project_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_groups
    ADD CONSTRAINT uq_region_text_consensus_groups_id_project_id UNIQUE (id, project_id);


--
-- Name: region_text_consensus_members uq_region_text_consensus_members_box; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_members
    ADD CONSTRAINT uq_region_text_consensus_members_box UNIQUE (project_id, item_id, box_id);


--
-- Name: region_text_runs uq_region_text_runs_id_project_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_runs
    ADD CONSTRAINT uq_region_text_runs_id_project_id UNIQUE (id, project_id);


--
-- Name: training_runs uq_training_runs_job_id; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_runs
    ADD CONSTRAINT uq_training_runs_job_id UNIQUE (job_id);


--
-- Name: ix_assist_loop_ledger_request; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_assist_loop_ledger_request ON public.assist_loop_budget_ledger USING btree (run_id, request_id);


--
-- Name: ix_assist_loop_nodes_decision; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_assist_loop_nodes_decision ON public.assist_loop_nodes USING btree (run_id, decision, level);


--
-- Name: ix_assist_loop_nodes_open; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_assist_loop_nodes_open ON public.assist_loop_nodes USING btree (run_id) WHERE (step <> 'done'::text);


--
-- Name: ix_assist_loop_proposals_request; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_assist_loop_proposals_request ON public.assist_loop_proposals USING btree (run_id, request_id);


--
-- Name: ix_assist_loop_runs_project; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_assist_loop_runs_project ON public.assist_loop_runs USING btree (project_id, created_at DESC);


--
-- Name: ix_cluster_memberships_cluster; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_cluster_memberships_cluster ON public.cluster_memberships USING btree (partition_id, cluster_uid);


--
-- Name: ix_cluster_partitions_recent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_cluster_partitions_recent ON public.cluster_partitions USING btree (project_id, target, level, created_at DESC);


--
-- Name: ix_config_activation_events_axis; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_config_activation_events_axis ON public.config_activation_events USING btree (project_id, axis, id DESC);


--
-- Name: ix_dataset_import_images_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_dataset_import_images_pending ON public.dataset_import_images USING btree (import_id) WHERE (status = 'pending'::text);


--
-- Name: ix_events_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_events_created ON public.events USING btree (created_at);


--
-- Name: ix_events_cursor; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_events_cursor ON public.events USING btree (project_id, xid, id);


--
-- Name: ix_events_cursor_global; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_events_cursor_global ON public.events USING btree (xid, id) WHERE (project_id IS NULL);


--
-- Name: ix_image_contents_imohash; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_image_contents_imohash ON public.image_contents USING btree (project_id, imohash);


--
-- Name: ix_ingest_source_files_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ingest_source_files_pending ON public.ingest_source_files USING btree (source_id, enqueued_at) WHERE (status = 'pending'::text);


--
-- Name: ix_jobs_interrupted; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_jobs_interrupted ON public.jobs USING btree (updated_at) WHERE ((status = 'interrupted'::text) AND auto_resume);


--
-- Name: ix_jobs_live; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_jobs_live ON public.jobs USING btree (status) WHERE (status = ANY (ARRAY['running'::text, 'cancelling'::text]));


--
-- Name: ix_jobs_project_recent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_jobs_project_recent ON public.jobs USING btree (project_id, kind, created_at DESC);


--
-- Name: ix_jobs_queued; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_jobs_queued ON public.jobs USING btree (queued_at) WHERE (status = 'queued'::text);


--
-- Name: ix_near_dup_edges_b; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_near_dup_edges_b ON public.near_dup_edges USING btree (project_id, level, space, b_id);


--
-- Name: ix_near_dup_groups_lineage; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_near_dup_groups_lineage ON public.near_dup_groups USING btree (lineage_id);


--
-- Name: ix_near_dup_groups_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_near_dup_groups_run ON public.near_dup_groups USING btree (run_id, purpose, member_count DESC);


--
-- Name: ix_near_dup_members_group; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_near_dup_members_group ON public.near_dup_members USING btree (group_id);


--
-- Name: ix_near_dup_pending_age; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_near_dup_pending_age ON public.near_dup_pending USING btree (project_id, level, enqueued_at);


--
-- Name: ix_near_dup_runs_recent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_near_dup_runs_recent ON public.near_dup_runs USING btree (project_id, level, started_at DESC);


--
-- Name: ix_project_classes_name_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_project_classes_name_key ON public.project_classes USING btree (project_id, name_key);


--
-- Name: ix_projects_updated_at_slug; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_projects_updated_at_slug ON public.projects USING btree (updated_at DESC, slug DESC);


--
-- Name: ix_region_text_consensus_groups_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_region_text_consensus_groups_run ON public.region_text_consensus_groups USING btree (run_id, project_id);


--
-- Name: ix_region_text_runs_project; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_region_text_runs_project ON public.region_text_runs USING btree (project_id, started_at DESC);


--
-- Name: ix_search_projection_tasks_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_search_projection_tasks_pending ON public.search_projection_tasks USING btree (created_at) WHERE (done_at IS NULL);


--
-- Name: uq_embedding_spaces_one_primary; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_embedding_spaces_one_primary ON public.embedding_spaces USING btree (project_id, target) WHERE (role = 'primary'::text);


--
-- Name: uq_holdout_current; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_holdout_current ON public.holdout_freezes USING btree (project_id) WHERE is_current;


--
-- Name: uq_image_contents_sha256; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_image_contents_sha256 ON public.image_contents USING btree (project_id, content_sha256) NULLS NOT DISTINCT WHERE (content_sha256 IS NOT NULL);


--
-- Name: uq_jobs_idempotency; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_jobs_idempotency ON public.jobs USING btree (project_id, kind, idempotency_key) NULLS NOT DISTINCT WHERE (idempotency_key IS NOT NULL);


--
-- Name: uq_jobs_one_active; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_jobs_one_active ON public.jobs USING btree (project_id, exclusive_group, scope) NULLS NOT DISTINCT WHERE ((exclusive_group IS NOT NULL) AND (status = ANY (ARRAY['cancelling'::text, 'paused'::text, 'queued'::text, 'running'::text])));


--
-- Name: uq_near_dup_members_rep; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_near_dup_members_rep ON public.near_dup_members USING btree (group_id) WHERE is_representative;


--
-- Name: uq_near_dup_runs_building; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_near_dup_runs_building ON public.near_dup_runs USING btree (project_id, level) NULLS NOT DISTINCT WHERE (status = 'building'::text);


--
-- Name: uq_near_dup_runs_current; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_near_dup_runs_current ON public.near_dup_runs USING btree (project_id, level) NULLS NOT DISTINCT WHERE (status = 'current'::text);


--
-- Name: uq_privacy_purges_kind_at; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_privacy_purges_kind_at ON public.privacy_purges USING btree (kind, at);


--
-- Name: uq_project_classes_active_name; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_project_classes_active_name ON public.project_classes USING btree (project_id, name_key) WHERE (NOT deprecated);


--
-- Name: uq_project_classes_hotkey; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_project_classes_hotkey ON public.project_classes USING btree (project_id, hotkey_letter) WHERE (hotkey_letter IS NOT NULL);


--
-- Name: uq_promotions_live; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_promotions_live ON public.promotions USING btree (triton_model_name) WHERE (superseded_at IS NULL);


--
-- Name: uq_region_text_runs_open_pass; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_region_text_runs_open_pass ON public.region_text_runs USING btree (project_id, reader, reader_version, profile_config_id, profile_revision) NULLS NOT DISTINCT WHERE ((trigger = 'region_pass'::text) AND (finished_at IS NULL));


--
-- Name: uq_upgrade_step_done; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_upgrade_step_done ON public.upgrade_runs USING btree (from_version, step) WHERE (status = 'completed'::text);


--
-- Name: cluster_calibrations cluster_calibrations_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER cluster_calibrations_immutable BEFORE DELETE OR UPDATE ON public.cluster_calibrations FOR EACH ROW EXECUTE FUNCTION public.cluster_calibrations_immutable();


--
-- Name: cluster_memberships cluster_children_project_from_parent; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER cluster_children_project_from_parent BEFORE INSERT ON public.cluster_memberships FOR EACH ROW EXECUTE FUNCTION public.cluster_children_project_from_parent();


--
-- Name: partition_clusters cluster_children_project_from_parent; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER cluster_children_project_from_parent BEFORE INSERT ON public.partition_clusters FOR EACH ROW EXECUTE FUNCTION public.cluster_children_project_from_parent();


--
-- Name: cluster_memberships cluster_children_ready_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER cluster_children_ready_immutable BEFORE DELETE OR UPDATE ON public.cluster_memberships FOR EACH ROW EXECUTE FUNCTION public.cluster_children_ready_immutable();


--
-- Name: partition_clusters cluster_children_ready_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER cluster_children_ready_immutable BEFORE DELETE OR UPDATE ON public.partition_clusters FOR EACH ROW EXECUTE FUNCTION public.cluster_children_ready_immutable();


--
-- Name: cluster_partitions cluster_partitions_ready_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER cluster_partitions_ready_immutable BEFORE DELETE OR UPDATE ON public.cluster_partitions FOR EACH ROW EXECUTE FUNCTION public.cluster_partitions_ready_immutable();


--
-- Name: config_revisions config_revisions_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER config_revisions_immutable BEFORE DELETE OR UPDATE ON public.config_revisions FOR EACH ROW EXECUTE FUNCTION public.config_revisions_immutable();


--
-- Name: embedding_model_versions embedding_model_versions_immutable; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER embedding_model_versions_immutable BEFORE UPDATE ON public.embedding_model_versions FOR EACH ROW EXECUTE FUNCTION public.embedding_model_versions_immutable();


--
-- Name: jobs jobs_status_transition_guard; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER jobs_status_transition_guard BEFORE UPDATE OF status ON public.jobs FOR EACH ROW WHEN ((old.status IS DISTINCT FROM new.status)) EXECUTE FUNCTION public.jobs_status_transition_guard();


--
-- Name: near_dup_groups near_dup_children_project_from_run; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER near_dup_children_project_from_run BEFORE INSERT ON public.near_dup_groups FOR EACH ROW EXECUTE FUNCTION public.near_dup_children_project_from_run();


--
-- Name: near_dup_members near_dup_children_project_from_run; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER near_dup_children_project_from_run BEFORE INSERT ON public.near_dup_members FOR EACH ROW EXECUTE FUNCTION public.near_dup_children_project_from_run();


--
-- Name: assist_loop_budget_ledger fk_assist_loop_budget_ledger_run_id_assist_loop_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_budget_ledger
    ADD CONSTRAINT fk_assist_loop_budget_ledger_run_id_assist_loop_runs FOREIGN KEY (run_id) REFERENCES public.assist_loop_runs(id) ON DELETE CASCADE;


--
-- Name: assist_loop_budgets fk_assist_loop_budgets_run_id_assist_loop_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_budgets
    ADD CONSTRAINT fk_assist_loop_budgets_run_id_assist_loop_runs FOREIGN KEY (run_id) REFERENCES public.assist_loop_runs(id) ON DELETE CASCADE;


--
-- Name: assist_loop_iterations fk_assist_loop_iterations_run_id_assist_loop_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_iterations
    ADD CONSTRAINT fk_assist_loop_iterations_run_id_assist_loop_runs FOREIGN KEY (run_id) REFERENCES public.assist_loop_runs(id) ON DELETE CASCADE;


--
-- Name: assist_loop_budget_ledger fk_assist_loop_ledger_budget; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_budget_ledger
    ADD CONSTRAINT fk_assist_loop_ledger_budget FOREIGN KEY (run_id, pool, unit) REFERENCES public.assist_loop_budgets(run_id, pool, unit) ON DELETE CASCADE;


--
-- Name: assist_loop_nodes fk_assist_loop_nodes_iteration; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_nodes
    ADD CONSTRAINT fk_assist_loop_nodes_iteration FOREIGN KEY (run_id, iteration) REFERENCES public.assist_loop_iterations(run_id, ordinal) ON DELETE CASCADE;


--
-- Name: assist_loop_nodes fk_assist_loop_nodes_label_op_id_label_operations; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_nodes
    ADD CONSTRAINT fk_assist_loop_nodes_label_op_id_label_operations FOREIGN KEY (label_op_id) REFERENCES public.label_operations(id) ON DELETE SET NULL;


--
-- Name: assist_loop_nodes fk_assist_loop_nodes_run_id_assist_loop_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_nodes
    ADD CONSTRAINT fk_assist_loop_nodes_run_id_assist_loop_runs FOREIGN KEY (run_id) REFERENCES public.assist_loop_runs(id) ON DELETE CASCADE;


--
-- Name: assist_loop_proposals fk_assist_loop_proposals_run_id_assist_loop_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_proposals
    ADD CONSTRAINT fk_assist_loop_proposals_run_id_assist_loop_runs FOREIGN KEY (run_id) REFERENCES public.assist_loop_runs(id) ON DELETE CASCADE;


--
-- Name: assist_loop_reviews fk_assist_loop_reviews_label_op_id_label_operations; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_reviews
    ADD CONSTRAINT fk_assist_loop_reviews_label_op_id_label_operations FOREIGN KEY (label_op_id) REFERENCES public.label_operations(id) ON DELETE SET NULL;


--
-- Name: assist_loop_reviews fk_assist_loop_reviews_run_id_assist_loop_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_reviews
    ADD CONSTRAINT fk_assist_loop_reviews_run_id_assist_loop_runs FOREIGN KEY (run_id) REFERENCES public.assist_loop_runs(id) ON DELETE CASCADE;


--
-- Name: assist_loop_runs fk_assist_loop_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_runs
    ADD CONSTRAINT fk_assist_loop_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id);


--
-- Name: assist_loop_runs fk_assist_loop_runs_preset; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_runs
    ADD CONSTRAINT fk_assist_loop_runs_preset FOREIGN KEY (preset_config_id, preset_revision) REFERENCES public.config_revisions(config_id, revision);


--
-- Name: assist_loop_runs fk_assist_loop_runs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_runs
    ADD CONSTRAINT fk_assist_loop_runs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: assist_loop_selections fk_assist_loop_selections_run_id_assist_loop_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assist_loop_selections
    ADD CONSTRAINT fk_assist_loop_selections_run_id_assist_loop_runs FOREIGN KEY (run_id) REFERENCES public.assist_loop_runs(id) ON DELETE CASCADE;


--
-- Name: bakeoff_runs fk_bakeoff_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bakeoff_runs
    ADD CONSTRAINT fk_bakeoff_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE CASCADE;


--
-- Name: bakeoff_runs fk_bakeoff_runs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bakeoff_runs
    ADD CONSTRAINT fk_bakeoff_runs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: class_registries fk_class_registries_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.class_registries
    ADD CONSTRAINT fk_class_registries_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: cluster_activation_events fk_cluster_activation_events_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_activation_events
    ADD CONSTRAINT fk_cluster_activation_events_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: cluster_activations fk_cluster_activations_pending_partition; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_activations
    ADD CONSTRAINT fk_cluster_activations_pending_partition FOREIGN KEY (pending_partition_id) REFERENCES public.cluster_partitions(id);


--
-- Name: cluster_activations fk_cluster_activations_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_activations
    ADD CONSTRAINT fk_cluster_activations_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: cluster_activations fk_cluster_activations_serving_partition; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_activations
    ADD CONSTRAINT fk_cluster_activations_serving_partition FOREIGN KEY (serving_partition_id) REFERENCES public.cluster_partitions(id);


--
-- Name: cluster_calibrations fk_cluster_calibrations_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_calibrations
    ADD CONSTRAINT fk_cluster_calibrations_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: cluster_identities fk_cluster_identities_first_partition; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_identities
    ADD CONSTRAINT fk_cluster_identities_first_partition FOREIGN KEY (first_partition_id) REFERENCES public.cluster_partitions(id) ON DELETE SET NULL;


--
-- Name: cluster_identities fk_cluster_identities_last_partition; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_identities
    ADD CONSTRAINT fk_cluster_identities_last_partition FOREIGN KEY (last_partition_id) REFERENCES public.cluster_partitions(id) ON DELETE SET NULL;


--
-- Name: cluster_identities fk_cluster_identities_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_identities
    ADD CONSTRAINT fk_cluster_identities_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: cluster_identities fk_cluster_identities_promoted_class; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_identities
    ADD CONSTRAINT fk_cluster_identities_promoted_class FOREIGN KEY (project_id, promoted_class_id) REFERENCES public.project_classes(project_id, class_id);


--
-- Name: cluster_memberships fk_cluster_memberships_partition_id_cluster_partitions; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_memberships
    ADD CONSTRAINT fk_cluster_memberships_partition_id_cluster_partitions FOREIGN KEY (partition_id) REFERENCES public.cluster_partitions(id) ON DELETE CASCADE;


--
-- Name: cluster_memberships fk_cluster_memberships_partition_project; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_memberships
    ADD CONSTRAINT fk_cluster_memberships_partition_project FOREIGN KEY (partition_id, project_id) REFERENCES public.cluster_partitions(id, project_id) ON DELETE CASCADE;


--
-- Name: cluster_partitions fk_cluster_partitions_calibration_id_cluster_calibrations; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_partitions
    ADD CONSTRAINT fk_cluster_partitions_calibration_id_cluster_calibrations FOREIGN KEY (calibration_id) REFERENCES public.cluster_calibrations(id);


--
-- Name: cluster_partitions fk_cluster_partitions_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_partitions
    ADD CONSTRAINT fk_cluster_partitions_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: cluster_partitions fk_cluster_partitions_parent_partition; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_partitions
    ADD CONSTRAINT fk_cluster_partitions_parent_partition FOREIGN KEY (parent_partition_id) REFERENCES public.cluster_partitions(id) ON DELETE SET NULL;


--
-- Name: cluster_partitions fk_cluster_partitions_profile; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_partitions
    ADD CONSTRAINT fk_cluster_partitions_profile FOREIGN KEY (profile_config_id, profile_revision) REFERENCES public.config_revisions(config_id, revision);


--
-- Name: cluster_partitions fk_cluster_partitions_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cluster_partitions
    ADD CONSTRAINT fk_cluster_partitions_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: combine_runs fk_combine_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine_runs
    ADD CONSTRAINT fk_combine_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE CASCADE;


--
-- Name: combine_runs fk_combine_runs_target_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine_runs
    ADD CONSTRAINT fk_combine_runs_target_project_id_projects FOREIGN KEY (target_project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: combine_sources fk_combine_sources_job_id_combine_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine_sources
    ADD CONSTRAINT fk_combine_sources_job_id_combine_runs FOREIGN KEY (job_id) REFERENCES public.combine_runs(job_id) ON DELETE CASCADE;


--
-- Name: combine_sources fk_combine_sources_source_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.combine_sources
    ADD CONSTRAINT fk_combine_sources_source_project_id_projects FOREIGN KEY (source_project_id) REFERENCES public.projects(id) ON DELETE SET NULL;


--
-- Name: config_activation_events fk_config_activation_events_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_activation_events
    ADD CONSTRAINT fk_config_activation_events_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: config_activations fk_config_activations_config_id_kind_config_documents; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_activations
    ADD CONSTRAINT fk_config_activations_config_id_kind_config_documents FOREIGN KEY (config_id, kind) REFERENCES public.config_documents(id, kind);


--
-- Name: config_activations fk_config_activations_config_id_revision_config_revisions; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_activations
    ADD CONSTRAINT fk_config_activations_config_id_revision_config_revisions FOREIGN KEY (config_id, revision) REFERENCES public.config_revisions(config_id, revision);


--
-- Name: config_activations fk_config_activations_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_activations
    ADD CONSTRAINT fk_config_activations_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: config_documents fk_config_documents_cloned_from_id_config_documents; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_documents
    ADD CONSTRAINT fk_config_documents_cloned_from_id_config_documents FOREIGN KEY (cloned_from_id) REFERENCES public.config_documents(id) ON DELETE SET NULL;


--
-- Name: config_documents fk_config_documents_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_documents
    ADD CONSTRAINT fk_config_documents_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: config_revisions fk_config_revisions_config_id_config_documents; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_revisions
    ADD CONSTRAINT fk_config_revisions_config_id_config_documents FOREIGN KEY (config_id) REFERENCES public.config_documents(id) ON DELETE CASCADE;


--
-- Name: config_runtime_reports fk_config_runtime_reports_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_runtime_reports
    ADD CONSTRAINT fk_config_runtime_reports_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: dataset_exports fk_dataset_exports_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_exports
    ADD CONSTRAINT fk_dataset_exports_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: dataset_exports fk_dataset_exports_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_exports
    ADD CONSTRAINT fk_dataset_exports_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: dataset_import_images fk_dataset_import_images_import_id_dataset_imports; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_import_images
    ADD CONSTRAINT fk_dataset_import_images_import_id_dataset_imports FOREIGN KEY (import_id) REFERENCES public.dataset_imports(id) ON DELETE CASCADE;


--
-- Name: dataset_imports fk_dataset_imports_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_imports
    ADD CONSTRAINT fk_dataset_imports_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: dataset_imports fk_dataset_imports_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_imports
    ADD CONSTRAINT fk_dataset_imports_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: dataset_imports fk_dataset_imports_undo_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_imports
    ADD CONSTRAINT fk_dataset_imports_undo_job_id_jobs FOREIGN KEY (undo_job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: dataset_imports fk_dataset_imports_upload_id_dataset_uploads; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_imports
    ADD CONSTRAINT fk_dataset_imports_upload_id_dataset_uploads FOREIGN KEY (upload_id) REFERENCES public.dataset_uploads(id) ON DELETE SET NULL;


--
-- Name: dataset_uploads fk_dataset_uploads_extract_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_uploads
    ADD CONSTRAINT fk_dataset_uploads_extract_job_id_jobs FOREIGN KEY (extract_job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: dataset_uploads fk_dataset_uploads_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_uploads
    ADD CONSTRAINT fk_dataset_uploads_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: embedding_evaluation_runs fk_embedding_evaluation_runs_baseline_space; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_evaluation_runs
    ADD CONSTRAINT fk_embedding_evaluation_runs_baseline_space FOREIGN KEY (baseline_space_id) REFERENCES public.embedding_spaces(id) ON DELETE CASCADE;


--
-- Name: embedding_evaluation_runs fk_embedding_evaluation_runs_candidate_space; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_evaluation_runs
    ADD CONSTRAINT fk_embedding_evaluation_runs_candidate_space FOREIGN KEY (candidate_space_id) REFERENCES public.embedding_spaces(id) ON DELETE CASCADE;


--
-- Name: embedding_evaluation_runs fk_embedding_evaluation_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_evaluation_runs
    ADD CONSTRAINT fk_embedding_evaluation_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: embedding_evaluation_runs fk_embedding_evaluation_runs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_evaluation_runs
    ADD CONSTRAINT fk_embedding_evaluation_runs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: embedding_reindex_runs fk_embedding_reindex_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_reindex_runs
    ADD CONSTRAINT fk_embedding_reindex_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id);


--
-- Name: embedding_reindex_runs fk_embedding_reindex_runs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_reindex_runs
    ADD CONSTRAINT fk_embedding_reindex_runs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: embedding_reindex_runs fk_embedding_reindex_runs_space_id_embedding_spaces; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_reindex_runs
    ADD CONSTRAINT fk_embedding_reindex_runs_space_id_embedding_spaces FOREIGN KEY (space_id) REFERENCES public.embedding_spaces(id) ON DELETE CASCADE;


--
-- Name: embedding_spaces fk_embedding_spaces_built_by_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_spaces
    ADD CONSTRAINT fk_embedding_spaces_built_by_job_id_jobs FOREIGN KEY (built_by_job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: embedding_spaces fk_embedding_spaces_model_version; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_spaces
    ADD CONSTRAINT fk_embedding_spaces_model_version FOREIGN KEY (model_version_id) REFERENCES public.embedding_model_versions(id) ON DELETE RESTRICT;


--
-- Name: embedding_spaces fk_embedding_spaces_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_spaces
    ADD CONSTRAINT fk_embedding_spaces_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: events fk_events_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT fk_events_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE CASCADE;


--
-- Name: events fk_events_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT fk_events_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: gpu_reservations fk_gpu_reservations_gpu_uuid_gpu_devices; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_reservations
    ADD CONSTRAINT fk_gpu_reservations_gpu_uuid_gpu_devices FOREIGN KEY (gpu_uuid) REFERENCES public.gpu_devices(gpu_uuid) ON DELETE RESTRICT;


--
-- Name: gpu_reservations fk_gpu_reservations_holder_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gpu_reservations
    ADD CONSTRAINT fk_gpu_reservations_holder_job_id_jobs FOREIGN KEY (holder_job_id) REFERENCES public.jobs(id) ON DELETE CASCADE;


--
-- Name: holdout_freezes fk_holdout_freezes_near_dup_image_run_id_near_dup_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_freezes
    ADD CONSTRAINT fk_holdout_freezes_near_dup_image_run_id_near_dup_runs FOREIGN KEY (near_dup_image_run_id) REFERENCES public.near_dup_runs(id) ON DELETE SET NULL;


--
-- Name: holdout_freezes fk_holdout_freezes_near_dup_item_run_id_near_dup_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_freezes
    ADD CONSTRAINT fk_holdout_freezes_near_dup_item_run_id_near_dup_runs FOREIGN KEY (near_dup_item_run_id) REFERENCES public.near_dup_runs(id) ON DELETE SET NULL;


--
-- Name: holdout_freezes fk_holdout_freezes_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_freezes
    ADD CONSTRAINT fk_holdout_freezes_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: holdout_near_dup_moves fk_holdout_near_dup_moves_freeze_id_holdout_freezes; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_near_dup_moves
    ADD CONSTRAINT fk_holdout_near_dup_moves_freeze_id_holdout_freezes FOREIGN KEY (freeze_id) REFERENCES public.holdout_freezes(id) ON DELETE CASCADE;


--
-- Name: holdout_near_dup_moves fk_holdout_near_dup_moves_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_near_dup_moves
    ADD CONSTRAINT fk_holdout_near_dup_moves_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: holdout_near_dup_moves fk_holdout_near_dup_moves_run_id_near_dup_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.holdout_near_dup_moves
    ADD CONSTRAINT fk_holdout_near_dup_moves_run_id_near_dup_runs FOREIGN KEY (run_id) REFERENCES public.near_dup_runs(id) ON DELETE SET NULL;


--
-- Name: image_contents fk_image_contents_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.image_contents
    ADD CONSTRAINT fk_image_contents_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: ingest_source_files fk_ingest_source_files_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingest_source_files
    ADD CONSTRAINT fk_ingest_source_files_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: ingest_source_files fk_ingest_source_files_source_id_ingest_sources; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingest_source_files
    ADD CONSTRAINT fk_ingest_source_files_source_id_ingest_sources FOREIGN KEY (source_id) REFERENCES public.ingest_sources(id) ON DELETE CASCADE;


--
-- Name: ingest_sources fk_ingest_sources_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingest_sources
    ADD CONSTRAINT fk_ingest_sources_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: job_attempts fk_job_attempts_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_attempts
    ADD CONSTRAINT fk_job_attempts_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE CASCADE;


--
-- Name: job_items fk_job_items_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_items
    ADD CONSTRAINT fk_job_items_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE CASCADE;


--
-- Name: jobs fk_jobs_parent_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT fk_jobs_parent_job_id_jobs FOREIGN KEY (parent_job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: jobs fk_jobs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT fk_jobs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: label_operation_members fk_label_operation_members_op_id_label_operations; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.label_operation_members
    ADD CONSTRAINT fk_label_operation_members_op_id_label_operations FOREIGN KEY (op_id) REFERENCES public.label_operations(id) ON DELETE CASCADE;


--
-- Name: label_operations fk_label_operations_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.label_operations
    ADD CONSTRAINT fk_label_operations_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: label_operations fk_label_operations_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.label_operations
    ADD CONSTRAINT fk_label_operations_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: model_engines fk_model_engines_build_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_engines
    ADD CONSTRAINT fk_model_engines_build_job_id_jobs FOREIGN KEY (build_job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: model_engines fk_model_engines_model_id_models; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_engines
    ADD CONSTRAINT fk_model_engines_model_id_models FOREIGN KEY (model_id) REFERENCES public.models(id) ON DELETE CASCADE;


--
-- Name: model_shares fk_model_shares_consumer_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_shares
    ADD CONSTRAINT fk_model_shares_consumer_project_id_projects FOREIGN KEY (consumer_project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: model_shares fk_model_shares_model_id_models; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_shares
    ADD CONSTRAINT fk_model_shares_model_id_models FOREIGN KEY (model_id) REFERENCES public.models(id) ON DELETE CASCADE;


--
-- Name: models fk_models_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT fk_models_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: models fk_models_training_run_id_training_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models
    ADD CONSTRAINT fk_models_training_run_id_training_runs FOREIGN KEY (training_run_id) REFERENCES public.training_runs(id) ON DELETE SET NULL;


--
-- Name: near_dup_decisions fk_near_dup_decisions_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_decisions
    ADD CONSTRAINT fk_near_dup_decisions_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: near_dup_edges fk_near_dup_edges_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_edges
    ADD CONSTRAINT fk_near_dup_edges_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: near_dup_groups fk_near_dup_groups_run_id_near_dup_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_groups
    ADD CONSTRAINT fk_near_dup_groups_run_id_near_dup_runs FOREIGN KEY (run_id) REFERENCES public.near_dup_runs(id) ON DELETE CASCADE;


--
-- Name: near_dup_groups fk_near_dup_groups_run_project; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_groups
    ADD CONSTRAINT fk_near_dup_groups_run_project FOREIGN KEY (run_id, project_id) REFERENCES public.near_dup_runs(id, project_id) ON DELETE CASCADE;


--
-- Name: near_dup_members fk_near_dup_members_group_id_near_dup_groups; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_members
    ADD CONSTRAINT fk_near_dup_members_group_id_near_dup_groups FOREIGN KEY (group_id) REFERENCES public.near_dup_groups(id) ON DELETE CASCADE;


--
-- Name: near_dup_members fk_near_dup_members_run_id_near_dup_runs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_members
    ADD CONSTRAINT fk_near_dup_members_run_id_near_dup_runs FOREIGN KEY (run_id) REFERENCES public.near_dup_runs(id) ON DELETE CASCADE;


--
-- Name: near_dup_members fk_near_dup_members_run_project; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_members
    ADD CONSTRAINT fk_near_dup_members_run_project FOREIGN KEY (run_id, project_id) REFERENCES public.near_dup_runs(id, project_id) ON DELETE CASCADE;


--
-- Name: near_dup_pending fk_near_dup_pending_claimed_by_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_pending
    ADD CONSTRAINT fk_near_dup_pending_claimed_by_job_id_jobs FOREIGN KEY (claimed_by_job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: near_dup_pending fk_near_dup_pending_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_pending
    ADD CONSTRAINT fk_near_dup_pending_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: near_dup_runs fk_near_dup_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_runs
    ADD CONSTRAINT fk_near_dup_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: near_dup_runs fk_near_dup_runs_parent_run; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_runs
    ADD CONSTRAINT fk_near_dup_runs_parent_run FOREIGN KEY (parent_run_id) REFERENCES public.near_dup_runs(id) ON DELETE SET NULL;


--
-- Name: near_dup_runs fk_near_dup_runs_profile; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_runs
    ADD CONSTRAINT fk_near_dup_runs_profile FOREIGN KEY (profile_config_id, profile_revision) REFERENCES public.config_revisions(config_id, revision);


--
-- Name: near_dup_runs fk_near_dup_runs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.near_dup_runs
    ADD CONSTRAINT fk_near_dup_runs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: partition_clusters fk_partition_clusters_cluster_uid_cluster_identities; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.partition_clusters
    ADD CONSTRAINT fk_partition_clusters_cluster_uid_cluster_identities FOREIGN KEY (cluster_uid) REFERENCES public.cluster_identities(cluster_uid) ON DELETE CASCADE;


--
-- Name: partition_clusters fk_partition_clusters_partition_id_cluster_partitions; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.partition_clusters
    ADD CONSTRAINT fk_partition_clusters_partition_id_cluster_partitions FOREIGN KEY (partition_id) REFERENCES public.cluster_partitions(id) ON DELETE CASCADE;


--
-- Name: partition_clusters fk_partition_clusters_partition_project; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.partition_clusters
    ADD CONSTRAINT fk_partition_clusters_partition_project FOREIGN KEY (partition_id, project_id) REFERENCES public.cluster_partitions(id, project_id) ON DELETE CASCADE;


--
-- Name: project_classes fk_project_classes_project_id_class_registries; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_classes
    ADD CONSTRAINT fk_project_classes_project_id_class_registries FOREIGN KEY (project_id) REFERENCES public.class_registries(project_id) ON DELETE CASCADE;


--
-- Name: project_classes fk_project_classes_project_id_merged_into_project_classes; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_classes
    ADD CONSTRAINT fk_project_classes_project_id_merged_into_project_classes FOREIGN KEY (project_id, merged_into) REFERENCES public.project_classes(project_id, class_id);


--
-- Name: project_indexes fk_project_indexes_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_indexes
    ADD CONSTRAINT fk_project_indexes_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: project_keymaps fk_project_keymaps_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_keymaps
    ADD CONSTRAINT fk_project_keymaps_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: project_settings fk_project_settings_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_settings
    ADD CONSTRAINT fk_project_settings_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: project_state fk_project_state_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_state
    ADD CONSTRAINT fk_project_state_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: projection_states fk_projection_states_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.projection_states
    ADD CONSTRAINT fk_projection_states_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: projection_states fk_projection_states_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.projection_states
    ADD CONSTRAINT fk_projection_states_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: projects fk_projects_slug_project_slugs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.projects
    ADD CONSTRAINT fk_projects_slug_project_slugs FOREIGN KEY (slug) REFERENCES public.project_slugs(slug);


--
-- Name: promotions fk_promotions_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.promotions
    ADD CONSTRAINT fk_promotions_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: promotions fk_promotions_model_id_models; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.promotions
    ADD CONSTRAINT fk_promotions_model_id_models FOREIGN KEY (model_id) REFERENCES public.models(id);


--
-- Name: promotions fk_promotions_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.promotions
    ADD CONSTRAINT fk_promotions_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: region_text_consensus_groups fk_region_text_consensus_groups_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_groups
    ADD CONSTRAINT fk_region_text_consensus_groups_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: region_text_consensus_groups fk_region_text_consensus_groups_run_project; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_groups
    ADD CONSTRAINT fk_region_text_consensus_groups_run_project FOREIGN KEY (run_id, project_id) REFERENCES public.region_text_runs(id, project_id) ON DELETE CASCADE;


--
-- Name: region_text_consensus_members fk_region_text_consensus_members_group_project; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_members
    ADD CONSTRAINT fk_region_text_consensus_members_group_project FOREIGN KEY (group_id, project_id) REFERENCES public.region_text_consensus_groups(id, project_id) ON DELETE CASCADE;


--
-- Name: region_text_consensus_members fk_region_text_consensus_members_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_consensus_members
    ADD CONSTRAINT fk_region_text_consensus_members_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: region_text_runs fk_region_text_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_runs
    ADD CONSTRAINT fk_region_text_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: region_text_runs fk_region_text_runs_profile; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_runs
    ADD CONSTRAINT fk_region_text_runs_profile FOREIGN KEY (profile_config_id, profile_revision) REFERENCES public.config_revisions(config_id, revision);


--
-- Name: region_text_runs fk_region_text_runs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.region_text_runs
    ADD CONSTRAINT fk_region_text_runs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: search_projection_tasks fk_search_projection_tasks_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_projection_tasks
    ADD CONSTRAINT fk_search_projection_tasks_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE SET NULL;


--
-- Name: search_projection_tasks fk_search_projection_tasks_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_projection_tasks
    ADD CONSTRAINT fk_search_projection_tasks_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: stream_worker_claims fk_stream_worker_claims_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stream_worker_claims
    ADD CONSTRAINT fk_stream_worker_claims_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: training_runs fk_training_runs_dataset_export_id_dataset_exports; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_runs
    ADD CONSTRAINT fk_training_runs_dataset_export_id_dataset_exports FOREIGN KEY (dataset_export_id) REFERENCES public.dataset_exports(id) ON DELETE SET NULL;


--
-- Name: training_runs fk_training_runs_job_id_jobs; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_runs
    ADD CONSTRAINT fk_training_runs_job_id_jobs FOREIGN KEY (job_id) REFERENCES public.jobs(id);


--
-- Name: training_runs fk_training_runs_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_runs
    ADD CONSTRAINT fk_training_runs_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: vlm_endpoint_probes fk_vlm_endpoint_probes_config_id_config_documents; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vlm_endpoint_probes
    ADD CONSTRAINT fk_vlm_endpoint_probes_config_id_config_documents FOREIGN KEY (config_id) REFERENCES public.config_documents(id) ON DELETE CASCADE;


--
-- Name: vlm_endpoint_probes fk_vlm_endpoint_probes_config_id_revision_config_revisions; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vlm_endpoint_probes
    ADD CONSTRAINT fk_vlm_endpoint_probes_config_id_revision_config_revisions FOREIGN KEY (config_id, revision) REFERENCES public.config_revisions(config_id, revision) ON DELETE CASCADE;


--
-- Name: vlm_usage_hourly fk_vlm_usage_hourly_endpoint_config_id_config_documents; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vlm_usage_hourly
    ADD CONSTRAINT fk_vlm_usage_hourly_endpoint_config_id_config_documents FOREIGN KEY (endpoint_config_id) REFERENCES public.config_documents(id) ON DELETE CASCADE;


--
-- Name: vlm_usage_hourly fk_vlm_usage_hourly_project_id_projects; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vlm_usage_hourly
    ADD CONSTRAINT fk_vlm_usage_hourly_project_id_projects FOREIGN KEY (project_id) REFERENCES public.projects(id) ON DELETE CASCADE;


--
-- Name: assist_loop_budget_ledger; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_budget_ledger ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_budgets; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_budgets ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_iterations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_iterations ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_nodes; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_nodes ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_proposals; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_proposals ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_reviews; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_reviews ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_selections; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.assist_loop_selections ENABLE ROW LEVEL SECURITY;

--
-- Name: bakeoff_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.bakeoff_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: class_registries; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.class_registries ENABLE ROW LEVEL SECURITY;

--
-- Name: cluster_activation_events; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cluster_activation_events ENABLE ROW LEVEL SECURITY;

--
-- Name: cluster_activations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cluster_activations ENABLE ROW LEVEL SECURITY;

--
-- Name: cluster_calibrations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cluster_calibrations ENABLE ROW LEVEL SECURITY;

--
-- Name: cluster_identities; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cluster_identities ENABLE ROW LEVEL SECURITY;

--
-- Name: cluster_memberships; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cluster_memberships ENABLE ROW LEVEL SECURITY;

--
-- Name: cluster_partitions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.cluster_partitions ENABLE ROW LEVEL SECURITY;

--
-- Name: combine_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.combine_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: combine_sources; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.combine_sources ENABLE ROW LEVEL SECURITY;

--
-- Name: config_activation_events; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.config_activation_events ENABLE ROW LEVEL SECURITY;

--
-- Name: config_activations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.config_activations ENABLE ROW LEVEL SECURITY;

--
-- Name: config_documents; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.config_documents ENABLE ROW LEVEL SECURITY;

--
-- Name: config_revisions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.config_revisions ENABLE ROW LEVEL SECURITY;

--
-- Name: config_runtime_reports; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.config_runtime_reports ENABLE ROW LEVEL SECURITY;

--
-- Name: dataset_exports; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.dataset_exports ENABLE ROW LEVEL SECURITY;

--
-- Name: dataset_import_images; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.dataset_import_images ENABLE ROW LEVEL SECURITY;

--
-- Name: dataset_imports; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.dataset_imports ENABLE ROW LEVEL SECURITY;

--
-- Name: dataset_uploads; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.dataset_uploads ENABLE ROW LEVEL SECURITY;

--
-- Name: embedding_evaluation_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.embedding_evaluation_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: embedding_reindex_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.embedding_reindex_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: embedding_spaces; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.embedding_spaces ENABLE ROW LEVEL SECURITY;

--
-- Name: events; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.events ENABLE ROW LEVEL SECURITY;

--
-- Name: gpu_reservations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.gpu_reservations ENABLE ROW LEVEL SECURITY;

--
-- Name: holdout_freezes; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.holdout_freezes ENABLE ROW LEVEL SECURITY;

--
-- Name: holdout_near_dup_moves; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.holdout_near_dup_moves ENABLE ROW LEVEL SECURITY;

--
-- Name: image_contents; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.image_contents ENABLE ROW LEVEL SECURITY;

--
-- Name: ingest_source_files; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.ingest_source_files ENABLE ROW LEVEL SECURITY;

--
-- Name: ingest_sources; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.ingest_sources ENABLE ROW LEVEL SECURITY;

--
-- Name: job_attempts; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.job_attempts ENABLE ROW LEVEL SECURITY;

--
-- Name: job_items; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.job_items ENABLE ROW LEVEL SECURITY;

--
-- Name: jobs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

--
-- Name: label_operation_members; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.label_operation_members ENABLE ROW LEVEL SECURITY;

--
-- Name: label_operations; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.label_operations ENABLE ROW LEVEL SECURITY;

--
-- Name: model_engines; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.model_engines ENABLE ROW LEVEL SECURITY;

--
-- Name: model_shares; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.model_shares ENABLE ROW LEVEL SECURITY;

--
-- Name: models; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;

--
-- Name: near_dup_decisions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_decisions ENABLE ROW LEVEL SECURITY;

--
-- Name: near_dup_edges; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_edges ENABLE ROW LEVEL SECURITY;

--
-- Name: near_dup_groups; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_groups ENABLE ROW LEVEL SECURITY;

--
-- Name: near_dup_members; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_members ENABLE ROW LEVEL SECURITY;

--
-- Name: near_dup_pending; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_pending ENABLE ROW LEVEL SECURITY;

--
-- Name: near_dup_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.near_dup_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: assist_loop_budget_ledger p_assist_loop_budget_ledger_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_budget_ledger_project ON public.assist_loop_budget_ledger USING ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_budget_ledger.run_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_budget_ledger.run_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: assist_loop_budgets p_assist_loop_budgets_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_budgets_project ON public.assist_loop_budgets USING ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_budgets.run_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_budgets.run_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: assist_loop_iterations p_assist_loop_iterations_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_iterations_project ON public.assist_loop_iterations USING ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_iterations.run_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_iterations.run_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: assist_loop_nodes p_assist_loop_nodes_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_nodes_project ON public.assist_loop_nodes USING ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_nodes.run_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_nodes.run_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: assist_loop_proposals p_assist_loop_proposals_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_proposals_project ON public.assist_loop_proposals USING ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_proposals.run_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_proposals.run_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: assist_loop_reviews p_assist_loop_reviews_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_reviews_project ON public.assist_loop_reviews USING ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_reviews.run_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_reviews.run_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: assist_loop_runs p_assist_loop_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_runs_project ON public.assist_loop_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: assist_loop_selections p_assist_loop_selections_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_assist_loop_selections_project ON public.assist_loop_selections USING ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_selections.run_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.assist_loop_runs x
  WHERE ((x.id = assist_loop_selections.run_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: bakeoff_runs p_bakeoff_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_bakeoff_runs_project ON public.bakeoff_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: class_registries p_class_registries_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_class_registries_project ON public.class_registries USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: cluster_activation_events p_cluster_activation_events_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_cluster_activation_events_project ON public.cluster_activation_events USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: cluster_activations p_cluster_activations_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_cluster_activations_project ON public.cluster_activations USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: cluster_calibrations p_cluster_calibrations_default_read; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_cluster_calibrations_default_read ON public.cluster_calibrations FOR SELECT USING (((project_id IS NULL) OR public.op_scope_allows(project_id)));


--
-- Name: cluster_calibrations p_cluster_calibrations_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_cluster_calibrations_project ON public.cluster_calibrations USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: cluster_identities p_cluster_identities_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_cluster_identities_project ON public.cluster_identities USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: cluster_memberships p_cluster_memberships_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_cluster_memberships_project ON public.cluster_memberships USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: cluster_partitions p_cluster_partitions_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_cluster_partitions_project ON public.cluster_partitions USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: combine_runs p_combine_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_combine_runs_project ON public.combine_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (target_project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (target_project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: combine_sources p_combine_sources_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_combine_sources_project ON public.combine_sources USING ((EXISTS ( SELECT 1
   FROM public.combine_runs x
  WHERE ((x.job_id = combine_sources.job_id) AND public.op_scope_allows(x.target_project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.combine_runs x
  WHERE ((x.job_id = combine_sources.job_id) AND public.op_scope_allows(x.target_project_id)))));


--
-- Name: config_activation_events p_config_activation_events_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_config_activation_events_project ON public.config_activation_events USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: config_activations p_config_activations_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_config_activations_project ON public.config_activations USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: config_documents p_config_documents_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_config_documents_project ON public.config_documents USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: config_revisions p_config_revisions_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_config_revisions_project ON public.config_revisions USING ((EXISTS ( SELECT 1
   FROM public.config_documents x
  WHERE ((x.id = config_revisions.config_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.config_documents x
  WHERE ((x.id = config_revisions.config_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: config_runtime_reports p_config_runtime_reports_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_config_runtime_reports_project ON public.config_runtime_reports USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: dataset_exports p_dataset_exports_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_dataset_exports_project ON public.dataset_exports USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: dataset_import_images p_dataset_import_images_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_dataset_import_images_project ON public.dataset_import_images USING ((EXISTS ( SELECT 1
   FROM public.dataset_imports x
  WHERE ((x.id = dataset_import_images.import_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.dataset_imports x
  WHERE ((x.id = dataset_import_images.import_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: dataset_imports p_dataset_imports_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_dataset_imports_project ON public.dataset_imports USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: dataset_uploads p_dataset_uploads_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_dataset_uploads_project ON public.dataset_uploads USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: embedding_evaluation_runs p_embedding_evaluation_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_embedding_evaluation_runs_project ON public.embedding_evaluation_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: embedding_reindex_runs p_embedding_reindex_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_embedding_reindex_runs_project ON public.embedding_reindex_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: embedding_spaces p_embedding_spaces_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_embedding_spaces_project ON public.embedding_spaces USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: events p_events_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_events_project ON public.events USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: gpu_reservations p_gpu_reservations_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_gpu_reservations_project ON public.gpu_reservations USING ((EXISTS ( SELECT 1
   FROM public.jobs x
  WHERE ((x.id = gpu_reservations.holder_job_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.jobs x
  WHERE ((x.id = gpu_reservations.holder_job_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: holdout_freezes p_holdout_freezes_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_holdout_freezes_project ON public.holdout_freezes USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: holdout_near_dup_moves p_holdout_near_dup_moves_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_holdout_near_dup_moves_project ON public.holdout_near_dup_moves USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: image_contents p_image_contents_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_image_contents_project ON public.image_contents USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: ingest_source_files p_ingest_source_files_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_ingest_source_files_project ON public.ingest_source_files USING ((EXISTS ( SELECT 1
   FROM public.ingest_sources x
  WHERE ((x.id = ingest_source_files.source_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.ingest_sources x
  WHERE ((x.id = ingest_source_files.source_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: ingest_sources p_ingest_sources_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_ingest_sources_project ON public.ingest_sources USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: job_attempts p_job_attempts_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_job_attempts_project ON public.job_attempts USING ((EXISTS ( SELECT 1
   FROM public.jobs x
  WHERE ((x.id = job_attempts.job_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.jobs x
  WHERE ((x.id = job_attempts.job_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: job_items p_job_items_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_job_items_project ON public.job_items USING ((EXISTS ( SELECT 1
   FROM public.jobs x
  WHERE ((x.id = job_items.job_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.jobs x
  WHERE ((x.id = job_items.job_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: jobs p_jobs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_jobs_project ON public.jobs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: label_operation_members p_label_operation_members_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_label_operation_members_project ON public.label_operation_members USING ((EXISTS ( SELECT 1
   FROM public.label_operations x
  WHERE ((x.id = label_operation_members.op_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.label_operations x
  WHERE ((x.id = label_operation_members.op_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: label_operations p_label_operations_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_label_operations_project ON public.label_operations USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: model_engines p_model_engines_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_model_engines_project ON public.model_engines USING (public.op_scope_allows(public.op_model_project(model_id))) WITH CHECK (public.op_scope_allows(public.op_model_project(model_id)));


--
-- Name: model_engines p_model_engines_shared; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_model_engines_shared ON public.model_engines FOR SELECT USING ((EXISTS ( SELECT 1
   FROM public.models m
  WHERE (m.id = model_engines.model_id))));


--
-- Name: model_shares p_model_shares_consumer; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_model_shares_consumer ON public.model_shares FOR SELECT USING (public.op_scope_allows(consumer_project_id));


--
-- Name: model_shares p_model_shares_owner; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_model_shares_owner ON public.model_shares USING (public.op_scope_allows(public.op_model_project(model_id))) WITH CHECK (public.op_scope_allows(public.op_model_project(model_id)));


--
-- Name: models p_models_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_models_project ON public.models USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: models p_models_shared; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_models_shared ON public.models FOR SELECT USING ((EXISTS ( SELECT 1
   FROM public.model_shares s
  WHERE ((s.model_id = models.id) AND public.op_scope_allows(s.consumer_project_id)))));


--
-- Name: near_dup_decisions p_near_dup_decisions_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_near_dup_decisions_project ON public.near_dup_decisions USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: near_dup_edges p_near_dup_edges_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_near_dup_edges_project ON public.near_dup_edges USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: near_dup_groups p_near_dup_groups_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_near_dup_groups_project ON public.near_dup_groups USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: near_dup_members p_near_dup_members_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_near_dup_members_project ON public.near_dup_members USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: near_dup_pending p_near_dup_pending_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_near_dup_pending_project ON public.near_dup_pending USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: near_dup_runs p_near_dup_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_near_dup_runs_project ON public.near_dup_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: partition_clusters p_partition_clusters_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_partition_clusters_project ON public.partition_clusters USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)) OR ((project_id IS NULL) AND ( SELECT public.op_scope_is_core() AS op_scope_is_core))));


--
-- Name: project_classes p_project_classes_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_project_classes_project ON public.project_classes USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: project_indexes p_project_indexes_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_project_indexes_project ON public.project_indexes USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: project_keymaps p_project_keymaps_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_project_keymaps_project ON public.project_keymaps USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: project_settings p_project_settings_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_project_settings_project ON public.project_settings USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: project_state p_project_state_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_project_state_project ON public.project_state USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: projection_states p_projection_states_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_projection_states_project ON public.projection_states USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: promotions p_promotions_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_promotions_project ON public.promotions USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: region_text_consensus_groups p_region_text_consensus_groups_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_region_text_consensus_groups_project ON public.region_text_consensus_groups USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: region_text_consensus_members p_region_text_consensus_members_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_region_text_consensus_members_project ON public.region_text_consensus_members USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: region_text_runs p_region_text_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_region_text_runs_project ON public.region_text_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: search_projection_tasks p_search_projection_tasks_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_search_projection_tasks_project ON public.search_projection_tasks USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: stream_worker_claims p_stream_worker_claims_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_stream_worker_claims_project ON public.stream_worker_claims USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: training_runs p_training_runs_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_training_runs_project ON public.training_runs USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: vlm_endpoint_probes p_vlm_endpoint_probes_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_vlm_endpoint_probes_project ON public.vlm_endpoint_probes USING ((EXISTS ( SELECT 1
   FROM public.config_documents x
  WHERE ((x.id = vlm_endpoint_probes.config_id) AND public.op_scope_allows(x.project_id))))) WITH CHECK ((EXISTS ( SELECT 1
   FROM public.config_documents x
  WHERE ((x.id = vlm_endpoint_probes.config_id) AND public.op_scope_allows(x.project_id)))));


--
-- Name: vlm_usage_hourly p_vlm_usage_hourly_project; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY p_vlm_usage_hourly_project ON public.vlm_usage_hourly USING ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project)))) WITH CHECK ((( SELECT public.op_scope_is_deployment() AS op_scope_is_deployment) OR (project_id = ( SELECT public.op_scope_project() AS op_scope_project))));


--
-- Name: partition_clusters; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.partition_clusters ENABLE ROW LEVEL SECURITY;

--
-- Name: project_classes; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.project_classes ENABLE ROW LEVEL SECURITY;

--
-- Name: project_indexes; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.project_indexes ENABLE ROW LEVEL SECURITY;

--
-- Name: project_keymaps; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.project_keymaps ENABLE ROW LEVEL SECURITY;

--
-- Name: project_settings; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.project_settings ENABLE ROW LEVEL SECURITY;

--
-- Name: project_state; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.project_state ENABLE ROW LEVEL SECURITY;

--
-- Name: projection_states; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.projection_states ENABLE ROW LEVEL SECURITY;

--
-- Name: promotions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.promotions ENABLE ROW LEVEL SECURITY;

--
-- Name: region_text_consensus_groups; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.region_text_consensus_groups ENABLE ROW LEVEL SECURITY;

--
-- Name: region_text_consensus_members; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.region_text_consensus_members ENABLE ROW LEVEL SECURITY;

--
-- Name: region_text_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.region_text_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: search_projection_tasks; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.search_projection_tasks ENABLE ROW LEVEL SECURITY;

--
-- Name: stream_worker_claims; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.stream_worker_claims ENABLE ROW LEVEL SECURITY;

--
-- Name: training_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.training_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: vlm_endpoint_probes; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.vlm_endpoint_probes ENABLE ROW LEVEL SECURITY;

--
-- Name: vlm_usage_hourly; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.vlm_usage_hourly ENABLE ROW LEVEL SECURITY;

--
-- PostgreSQL database dump complete
--

\unrestrict opschemahead
